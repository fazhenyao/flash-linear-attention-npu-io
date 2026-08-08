#!/usr/bin/env python3
"""Outbound-only VPN Relay agent for Cloudflare performance jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from .perf_runner import execute, load_config, runner_status
except ImportError:
    from perf_runner import execute, load_config, runner_status  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "https://flash-linear-attention-npu-io-fazhenyao.fazhenyao.workers.dev"
FINAL_STATES = {"succeeded", "failed", "canceled", "orphaned"}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class AgentConfig:
    api_base: str
    token: str
    runner_id: str
    runner_name: str
    poll_min_seconds: int
    poll_max_seconds: int
    error_backoff_max_seconds: int
    heartbeat_seconds: int
    max_concurrency: int
    state_dir: Path
    retention_days: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        token = os.environ.get("RUNNER_TOKEN", "").strip()
        if not token:
            raise ValueError("RUNNER_TOKEN 未配置")
        runner_id = os.environ.get("RUNNER_ID", "vpn-runner-01").strip()
        if not runner_id:
            raise ValueError("RUNNER_ID 未配置")
        state_dir = Path(os.environ.get("RUNNER_STATE_DIR", ROOT / "data" / "runner-state"))
        if not state_dir.is_absolute():
            state_dir = ROOT / state_dir
        return cls(
            api_base=os.environ.get("RUNNER_API_BASE", DEFAULT_API_BASE).strip().rstrip("/"),
            token=token,
            runner_id=runner_id,
            runner_name=os.environ.get("RUNNER_NAME", runner_id).strip() or runner_id,
            poll_min_seconds=env_int("RUNNER_POLL_MIN_SECONDS", 2, 1),
            poll_max_seconds=env_int("RUNNER_POLL_MAX_SECONDS", 30, 2),
            error_backoff_max_seconds=env_int("RUNNER_ERROR_BACKOFF_MAX_SECONDS", 120, 5),
            heartbeat_seconds=env_int("RUNNER_HEARTBEAT_SECONDS", 15, 5),
            max_concurrency=env_int("RUNNER_MAX_CONCURRENCY", 1, 1),
            state_dir=state_dir,
            retention_days=env_int("RUNNER_ARTIFACT_RETENTION_DAYS", 30, 1),
        )


class RunnerApi:
    def __init__(self, config: AgentConfig):
        self.config = config

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.api_base + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": f"fla-vpn-runner/{self.config.runner_id}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Runner API HTTP {exc.code}: {detail[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Runner API 不可达：{exc.reason}") from exc


class JobHeartbeat(threading.Thread):
    def __init__(self, agent: "RunnerAgent", job: dict[str, Any]):
        super().__init__(name=f"heartbeat-{job['id']}", daemon=True)
        self.agent = agent
        self.job = job
        self.stop_event = threading.Event()
        self.cancel_requested = threading.Event()
        self.last_error = ""

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.wait(self.agent.config.heartbeat_seconds):
            try:
                health = self.agent.health()
                self.agent.send_runner_heartbeat(health, current_jobs=1)
                response = self.agent.api.post(
                    f"/api/runner/jobs/{self.job['id']}/heartbeat",
                    self.agent.job_auth(self.job, {
                        "state": "running" if health["npu_reachable"] else "disconnected",
                        "message": "采集执行中" if health["npu_reachable"] else "VPN 或 NPU SSH 暂不可达",
                    }),
                )
                if response.get("cancel_requested"):
                    self.cancel_requested.set()
            except Exception as exc:  # heartbeat failure must not hide the execution result
                self.last_error = str(exc)


class RunnerAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.api = RunnerApi(config)
        self.stop_event = threading.Event()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "jobs").mkdir(parents=True, exist_ok=True)

    def stop(self, *_args: object) -> None:
        self.stop_event.set()

    def capabilities(self) -> dict[str, Any]:
        status = runner_status()
        device = status.get("npu_device")
        chip = status.get("chip")
        return {
            "mode": status.get("mode"),
            "chip": chip,
            "chips": [chip] if chip else [],
            "device": device,
            "devices": [device] if device is not None else [],
            "prof_tools": status.get("prof_tools") or [],
            "op_warm_up": status.get("op_warm_up"),
            "op_launch_count": status.get("op_launch_count"),
            "max_concurrency": self.config.max_concurrency,
            "agent_version": "1.0.0",
        }

    def health(self) -> dict[str, Any]:
        status = runner_status()
        if not status.get("enabled"):
            return {
                "vpn_connected": False,
                "npu_reachable": False,
                "last_error": status.get("error") or "执行环境未配置",
            }
        perf_config = load_config()
        if perf_config.mode == "local":
            return {"vpn_connected": True, "npu_reachable": True, "last_error": ""}
        port = int(perf_config.ssh_port or "22")
        try:
            with socket.create_connection((perf_config.ssh_host, port), timeout=5):
                return {"vpn_connected": True, "npu_reachable": True, "last_error": ""}
        except OSError as exc:
            return {
                "vpn_connected": False,
                "npu_reachable": False,
                "last_error": f"NPU SSH 不可达：{exc}",
            }

    def runner_payload(self, health: dict[str, Any], current_jobs: int = 0) -> dict[str, Any]:
        return {
            "runner_id": self.config.runner_id,
            "name": self.config.runner_name,
            "capabilities": self.capabilities(),
            "vpn_connected": bool(health.get("vpn_connected")),
            "npu_reachable": bool(health.get("npu_reachable")),
            "current_jobs": current_jobs,
            "last_error": str(health.get("last_error") or "")[:1000],
        }

    def send_runner_heartbeat(self, health: dict[str, Any], current_jobs: int = 0) -> dict[str, Any]:
        return self.api.post("/api/runner/heartbeat", self.runner_payload(health, current_jobs))

    def register(self) -> dict[str, Any]:
        health = self.health()
        return self.api.post("/api/runner/register", self.runner_payload(health))

    def claim(self, health: dict[str, Any]) -> dict[str, Any] | None:
        response = self.api.post("/api/runner/jobs/claim", self.runner_payload(health))
        return response.get("job")

    def job_auth(self, job: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "runner_id": self.config.runner_id,
            "attempt_id": job["attempt_id"],
            "lease_token": job["lease_token"],
            **(payload or {}),
        }

    def job_state_path(self, job_id: str) -> Path:
        return self.config.state_dir / "jobs" / f"{job_id}.json"

    def save_job_state(self, job: dict[str, Any], state: str, **extra: Any) -> None:
        record = {
            "job_id": job["id"],
            "attempt_id": job.get("attempt_id"),
            "runner_id": self.config.runner_id,
            "state": state,
            "updated_at": utc_now(),
            **extra,
        }
        path = self.job_state_path(job["id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def run_job(self, job: dict[str, Any]) -> None:
        request = dict(job.get("request") or {})
        self.save_job_state(job, "claimed", request=request)
        self.api.post(
            f"/api/runner/jobs/{job['id']}/started",
            self.job_auth(job, {
                "remote_execution_id": f"{self.config.runner_id}:{job['id']}",
                "message": "Relay 已开始执行采集任务",
            }),
        )
        heartbeat = JobHeartbeat(self, job)
        heartbeat.start()
        try:
            self.save_job_state(job, "running", request=request)
            result = execute(request)
            artifacts, local_artifacts = self.build_artifacts(job, result)
            self.save_job_state(
                job,
                "completed",
                request=request,
                artifacts=local_artifacts,
                message=result.get("message", ""),
            )
            if heartbeat.cancel_requested.is_set():
                self.api.post(
                    f"/api/runner/jobs/{job['id']}/fail",
                    self.job_auth(job, {"canceled": True, "message": "采集完成前收到取消请求"}),
                )
                return
            snapshot = result.get("snapshot") or {}
            self.api.post(
                f"/api/runner/jobs/{job['id']}/complete",
                self.job_auth(job, {
                    "exit_code": 0,
                    "message": result.get("message") or "性能采集完成",
                    "command": result.get("command") or "",
                    "environment": self.environment_summary(),
                    "metrics": extract_snapshot_metrics(snapshot),
                    "snapshot": snapshot,
                    "perf_data": result.get("data") or {},
                    "result": {
                        "message": result.get("message") or "",
                        "prof_tool": result.get("prof_tool") or request.get("prof_tool"),
                        "prof_source": result.get("prof_source") or snapshot.get("prof_source"),
                    },
                    "artifacts": artifacts,
                }),
            )
        except Exception as exc:
            self.save_job_state(job, "failed", request=request, error=str(exc))
            self.api.post(
                f"/api/runner/jobs/{job['id']}/fail",
                self.job_auth(job, {
                    "message": str(exc),
                    "error_type": exc.__class__.__name__,
                    "canceled": heartbeat.cancel_requested.is_set(),
                }),
            )
        finally:
            heartbeat.stop()
            heartbeat.join(timeout=2)
            try:
                self.send_runner_heartbeat(self.health(), current_jobs=0)
            except Exception:
                pass

    def environment_summary(self) -> dict[str, Any]:
        status = runner_status()
        return {
            "agent_version": "1.0.0",
            "runner_id": self.config.runner_id,
            "mode": status.get("mode"),
            "chip": status.get("chip"),
            "device": status.get("npu_device"),
            "soc_version": status.get("soc_version"),
        }

    def build_artifacts(self, job: dict[str, Any], result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw = str(result.get("prof_dir") or "").strip()
        if not raw:
            return [], []
        path = Path(raw).resolve()
        if not path.exists():
            return [], []
        size, digest = directory_digest(path)
        expires_at = (datetime.now(timezone.utc) + timedelta(days=self.config.retention_days)).isoformat()
        object_key = f"relay://{self.config.runner_id}/{job['id']}/{path.name}"
        public = {
            "type": "prof_directory",
            "object_key": object_key,
            "filename": path.name,
            "content_type": "application/x-directory",
            "size_bytes": size,
            "sha256": digest,
            "expires_at": expires_at,
        }
        local = {**public, "local_path": str(path), "deleted": False}
        return [public], [local]

    def cleanup_expired_artifacts(self) -> int:
        deleted = 0
        now = datetime.now(timezone.utc)
        roots = artifact_roots()
        for state_path in (self.config.state_dir / "jobs").glob("*.json"):
            try:
                record = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            changed = False
            for artifact in record.get("artifacts") or []:
                if artifact.get("deleted") or not artifact.get("local_path"):
                    continue
                expires_at = parse_iso(artifact.get("expires_at"))
                path = Path(artifact["local_path"]).resolve()
                if expires_at is None or expires_at > now or not path_within_roots(path, roots):
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                artifact["deleted"] = True
                artifact["deleted_at"] = utc_now()
                changed = True
                deleted += 1
            if changed:
                state_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return deleted

    def run_once(self) -> bool:
        self.cleanup_expired_artifacts()
        health = self.health()
        self.send_runner_heartbeat(health)
        if not health["vpn_connected"] or not health["npu_reachable"]:
            return False
        job = self.claim(health)
        if not job:
            return False
        self.run_job(job)
        return True

    def run_forever(self) -> None:
        self.register()
        delay = self.config.poll_min_seconds
        while not self.stop_event.is_set():
            try:
                worked = self.run_once()
                delay = self.config.poll_min_seconds if worked else min(self.config.poll_max_seconds, delay + 2)
            except Exception as exc:
                print(f"[runner] {exc}", flush=True)
                delay = min(self.config.error_backoff_max_seconds, max(delay * 2, 5))
            self.stop_event.wait(delay)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def directory_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return size, digest.hexdigest()


def extract_snapshot_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    keys = ["total_time_ms", "duration_us", "mfu", "mbu", "bandwidth_gbps", "tflops"]
    metrics = {key: snapshot[key] for key in keys if snapshot.get(key) is not None}
    if isinstance(snapshot.get("metrics"), dict):
        metrics.update(snapshot["metrics"])
    return metrics


def artifact_roots() -> list[Path]:
    configured = os.environ.get("RUNNER_ARTIFACT_ROOTS", "data/prof_gdr;data/prof_op")
    roots = []
    for raw in configured.split(";"):
        if not raw.strip():
            continue
        path = Path(raw.strip())
        roots.append((path if path.is_absolute() else ROOT / path).resolve())
    return roots


def path_within_roots(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloudflare D1 performance queue VPN Relay")
    parser.add_argument("--once", action="store_true", help="执行一次 heartbeat/claim 后退出")
    parser.add_argument("--check", action="store_true", help="仅打印本地健康状态，不连接 Worker")
    args = parser.parse_args()
    config = AgentConfig.from_env()
    agent = RunnerAgent(config)
    signal.signal(signal.SIGINT, agent.stop)
    signal.signal(signal.SIGTERM, agent.stop)
    if args.check:
        print(json.dumps({"runner_id": config.runner_id, **agent.health(), "capabilities": agent.capabilities()}, ensure_ascii=False, indent=2))
        return 0
    if args.once:
        agent.run_once()
        return 0
    agent.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
