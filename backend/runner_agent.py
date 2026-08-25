#!/usr/bin/env python3
"""Outbound-only VPN Relay agent for Cloudflare performance jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import websocket as websocket_client
except ImportError:  # The feature flag falls back to HTTPS polling until installed.
    websocket_client = None

try:
    from .perf_runner import (
        acknowledge_persistent_build_install,
        cancel_persistent_build_install,
        collect_npu_device_status,
        execute,
        execution_environment_defaults,
        list_remote_source_branches,
        load_config,
        persistent_build_handle,
        persistent_build_result,
        poll_persistent_build_install,
        RemoteBuildOrphanedError,
        RemoteConnectionError,
        runner_status,
        start_persistent_build_install,
    )
except ImportError:
    from perf_runner import (  # type: ignore
        acknowledge_persistent_build_install,
        cancel_persistent_build_install,
        collect_npu_device_status,
        execute,
        execution_environment_defaults,
        list_remote_source_branches,
        load_config,
        persistent_build_handle,
        persistent_build_result,
        poll_persistent_build_install,
        RemoteBuildOrphanedError,
        RemoteConnectionError,
        runner_status,
        start_persistent_build_install,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "https://flash-linear-attention-npu-io-fazhenyao.fazhenyao.workers.dev"
FINAL_STATES = {"succeeded", "failed", "canceled", "orphaned"}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    notification_enabled: bool
    reconcile_seconds: int
    ws_reconnect_min_seconds: int
    ws_reconnect_max_seconds: int
    ws_connect_timeout_seconds: int
    max_concurrency: int
    state_dir: Path
    retention_days: int
    upload_part_bytes: int
    upload_concurrency: int
    npu_status_interval_seconds: int
    npu_status_timeout_seconds: int
    npu_device_count: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        token = os.environ.get("RUNNER_TOKEN", "").strip()
        if not token:
            raise ValueError("RUNNER_TOKEN 未配置")
        runner_id = os.environ.get("RUNNER_ID", "vpn-runner-01").strip()
        if not runner_id:
            raise ValueError("RUNNER_ID 未配置")
        ws_reconnect_min_seconds = env_int("RUNNER_WS_RECONNECT_MIN_SECONDS", 1, 1)
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
            notification_enabled=env_bool("RUNNER_NOTIFICATION_ENABLED", False),
            reconcile_seconds=env_int("RUNNER_RECONCILE_SECONDS", 300, 30),
            ws_reconnect_min_seconds=ws_reconnect_min_seconds,
            ws_reconnect_max_seconds=max(
                ws_reconnect_min_seconds,
                env_int("RUNNER_WS_RECONNECT_MAX_SECONDS", 30, 1),
            ),
            ws_connect_timeout_seconds=env_int("RUNNER_WS_CONNECT_TIMEOUT_SECONDS", 15, 5),
            max_concurrency=env_int("RUNNER_MAX_CONCURRENCY", 1, 1),
            state_dir=state_dir,
            retention_days=env_int("RUNNER_ARTIFACT_RETENTION_DAYS", 30, 1),
            upload_part_bytes=env_int("RUNNER_R2_PART_MIB", 32, 5) * 1024 * 1024,
            upload_concurrency=env_int("RUNNER_UPLOAD_CONCURRENCY", 1, 1),
            npu_status_interval_seconds=env_int("RUNNER_NPU_STATUS_INTERVAL_SECONDS", 1800, 10),
            npu_status_timeout_seconds=env_int("RUNNER_NPU_STATUS_TIMEOUT_SECONDS", 60, 10),
            npu_device_count=env_int("RUNNER_NPU_DEVICE_COUNT", 8, 1),
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

    def put(self, path: str, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.api_base + path,
            data=body,
            method="PUT",
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(body)),
                "User-Agent": f"fla-vpn-runner/{self.config.runner_id}",
                **headers,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Runner upload API HTTP {exc.code}: {detail[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Runner upload API 不可达：{exc.reason}") from exc


def runner_events_url(api_base: str, runner_id: str) -> str:
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("RUNNER_API_BASE 必须是 HTTP(S) URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/api/runner/events"
    query = urllib.parse.urlencode({"runner_id": runner_id})
    return urllib.parse.urlunsplit((scheme, parsed.netloc, path, query, ""))


class RunnerNotificationClient:
    def __init__(self, agent: "RunnerAgent", connector: Any = None):
        self.agent = agent
        self.config = agent.config
        self._connector = connector or (
            websocket_client.create_connection if websocket_client is not None else None
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        self._socket_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "mode": "websocket" if getattr(self.config, "notification_enabled", False) else "polling",
            "enabled": bool(getattr(self.config, "notification_enabled", False)),
            "available": self._connector is not None,
            "connected": False,
            "connected_at": None,
            "last_event_at": None,
            "last_error": "",
            "reconnect_count": 0,
            "claim_wakeups": 0,
            "reconcile_last_at": None,
        }

    @property
    def active(self) -> bool:
        return bool(getattr(self.config, "notification_enabled", False) and self._connector)

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def _update_status(self, **values: Any) -> None:
        with self._status_lock:
            self._status.update(values)

    def mark_reconcile(self) -> None:
        self._update_status(reconcile_last_at=utc_now())

    def start(self) -> bool:
        if not getattr(self.config, "notification_enabled", False):
            return False
        if self._connector is None:
            message = "websocket-client 未安装，已回退到 HTTPS 轮询"
            self._update_status(mode="polling_fallback", available=False, last_error=message)
            print(f"[runner] {message}", flush=True)
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._thread = threading.Thread(
            target=self._run,
            name=f"runner-events-{self.config.runner_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        with self._socket_lock:
            active_socket = self._socket
        if active_socket is not None:
            try:
                active_socket.close()
            except Exception:
                pass
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _connect(self) -> Any:
        return self._connector(
            runner_events_url(self.config.api_base, self.config.runner_id),
            header={
                "Authorization": f"Bearer {self.config.token}",
                "User-Agent": f"fla-vpn-runner/{self.config.runner_id}",
            },
            timeout=self.config.ws_connect_timeout_seconds,
            enable_multithread=True,
        )

    def _handle_message(self, raw_message: Any) -> None:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8", errors="replace")
        try:
            message = json.loads(str(raw_message))
        except (TypeError, ValueError):
            return
        if (
            message.get("version") != 1
            or message.get("type") != "job_available"
            or message.get("runner_id") != self.config.runner_id
        ):
            return
        with self._status_lock:
            self._status["last_event_at"] = utc_now()
            self._status["claim_wakeups"] += 1
            self._status["last_error"] = ""
        self.agent._dispatch_event.set()

    @staticmethod
    def _is_timeout(error: Exception) -> bool:
        return isinstance(error, socket.timeout) or error.__class__.__name__ == "WebSocketTimeoutException"

    def _run(self) -> None:
        reconnect_delay = self.config.ws_reconnect_min_seconds
        connected_once = False
        while not self._stop_event.is_set() and not self.agent.stop_event.is_set():
            active_socket = None
            try:
                active_socket = self._connect()
                with self._socket_lock:
                    self._socket = active_socket
                if hasattr(active_socket, "settimeout"):
                    active_socket.settimeout(30)
                if connected_once:
                    with self._status_lock:
                        self._status["reconnect_count"] += 1
                connected_once = True
                reconnect_delay = self.config.ws_reconnect_min_seconds
                self._update_status(
                    mode="websocket",
                    available=True,
                    connected=True,
                    connected_at=utc_now(),
                    last_error="",
                )
                self.agent._dispatch_event.set()
                while not self._stop_event.is_set() and not self.agent.stop_event.is_set():
                    try:
                        message = active_socket.recv()
                        if message is None or message == "":
                            raise ConnectionError("WebSocket 已关闭")
                        self._handle_message(message)
                    except Exception as exc:
                        if self._is_timeout(exc):
                            try:
                                active_socket.ping()
                                continue
                            except Exception as ping_error:
                                raise ConnectionError(f"WebSocket Ping 失败：{ping_error}") from ping_error
                        raise
            except Exception as exc:
                if not self._stop_event.is_set() and not self.agent.stop_event.is_set():
                    self._update_status(
                        connected=False,
                        last_error=str(exc)[:1000],
                    )
            finally:
                with self._socket_lock:
                    if self._socket is active_socket:
                        self._socket = None
                if active_socket is not None:
                    try:
                        active_socket.close()
                    except Exception:
                        pass
                self._update_status(connected=False)
            if self._stop_event.is_set() or self.agent.stop_event.is_set():
                break
            jitter = random.uniform(0, min(1.0, reconnect_delay * 0.2))
            if self._stop_event.wait(reconnect_delay + jitter):
                break
            reconnect_delay = min(self.config.ws_reconnect_max_seconds, reconnect_delay * 2)


class JobHeartbeat(threading.Thread):
    def __init__(self, agent: "RunnerAgent", job: dict[str, Any]):
        super().__init__(name=f"heartbeat-{job['id']}", daemon=True)
        self.agent = agent
        self.job = job
        self.stop_event = threading.Event()
        self.cancel_requested = threading.Event()
        self.last_error = ""
        self.remote_state = ""

    def set_remote_state(self, state: str) -> None:
        self.remote_state = str(state or "")

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.wait(self.agent.config.heartbeat_seconds):
            try:
                health = self.agent.health()
                self.agent.send_runner_heartbeat(health)
                task_type = str((self.job.get("request") or {}).get("task_type") or "profile")
                if task_type == "build_install" and self.remote_state:
                    heartbeat_state = "disconnected" if not health["npu_reachable"] else "running"
                    message = (
                        f"远端编译阶段：{self.remote_state}"
                        if health["npu_reachable"]
                        else f"VPN 或 NPU SSH 暂不可达，最后远端阶段：{self.remote_state}"
                    )
                else:
                    heartbeat_state = "running" if health["npu_reachable"] else "disconnected"
                    message = "测试执行中" if health["npu_reachable"] else "VPN 或 NPU SSH 暂不可达"
                response = self.agent.api.post(
                    f"/api/runner/jobs/{self.job['id']}/heartbeat",
                    self.agent.job_auth(self.job, {
                        "state": heartbeat_state,
                        "remote_state": self.remote_state or None,
                        "message": message,
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
        self._dispatch_event = threading.Event()
        self._active_jobs_lock = threading.Lock()
        self._active_job_ids: set[str] = set()
        self.current_jobs = 0
        self._futures_lock = threading.Lock()
        self._futures: dict[Future[None], str] = {}
        max_concurrency = max(1, int(getattr(config, "max_concurrency", 1)))
        upload_concurrency = max(1, int(getattr(config, "upload_concurrency", 1)))
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix=f"runner-{config.runner_id}",
        )
        self._build_slot = threading.Lock()
        self._upload_slots = threading.BoundedSemaphore(upload_concurrency)
        self.notification = RunnerNotificationClient(self)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self._npu_status_lock = threading.Lock()
        self._npu_status_refreshing = False
        self._npu_status_pending_refresh_id: str | None = None
        self._npu_status_checked_at = 0.0
        self._npu_status = {
            "updated_at": None,
            "checked_at": None,
            "devices": [],
            "error": "等待 Relay 首次查询",
        }
        self._source_branches_lock = threading.Lock()
        self._source_branches_refreshing = False
        self._source_branches_pending: tuple[str, str] | None = None
        self._source_branches = self._load_source_branches_cache()

    def stop(self, *_args: object) -> None:
        self.stop_event.set()
        self._dispatch_event.set()
        notification = getattr(self, "notification", None)
        if notification is not None:
            notification.stop(wait=False)

    def active_job_count(self) -> int:
        lock = getattr(self, "_active_jobs_lock", None)
        if lock is None:
            return int(getattr(self, "current_jobs", 0))
        with lock:
            return len(self._active_job_ids)

    def _mark_job_active(self, job_id: str) -> None:
        lock = getattr(self, "_active_jobs_lock", None)
        if lock is None:
            self.current_jobs = int(getattr(self, "current_jobs", 0)) + 1
            return
        with lock:
            self._active_job_ids.add(job_id)
            self.current_jobs = len(self._active_job_ids)

    def _mark_job_inactive(self, job_id: str) -> None:
        lock = getattr(self, "_active_jobs_lock", None)
        if lock is None:
            self.current_jobs = max(0, int(getattr(self, "current_jobs", 1)) - 1)
            return
        with lock:
            self._active_job_ids.discard(job_id)
            self.current_jobs = len(self._active_job_ids)

    def available_slots(self) -> int:
        lock = getattr(self, "_futures_lock", None)
        if lock is None:
            return max(0, self.config.max_concurrency - self.active_job_count())
        with lock:
            return max(0, self.config.max_concurrency - len(self._futures))

    def submit_job(self, job: dict[str, Any], *, resume: bool = False) -> bool:
        if self.stop_event.is_set():
            return False
        with self._futures_lock:
            if len(self._futures) >= self.config.max_concurrency:
                return False
            future = self._executor.submit(self.run_job, job, resume=resume)
            self._futures[future] = str(job["id"])
        future.add_done_callback(self._job_finished)
        return True

    def _job_finished(self, future: Future[None]) -> None:
        with self._futures_lock:
            job_id = self._futures.pop(future, "unknown")
        try:
            future.result()
        except Exception as exc:
            print(f"[runner] 任务线程 {job_id} 异常退出：{exc}", flush=True)
        self._dispatch_event.set()

    def capabilities(self, *, refresh_npu: bool = True) -> dict[str, Any]:
        status = runner_status()
        perf_config = load_config()
        device = status.get("npu_device")
        chip = status.get("chip")
        npu_status = self.npu_status(refresh=refresh_npu)
        return {
            "mode": status.get("mode"),
            "chip": chip,
            "chips": [chip] if chip else [],
            "device": device,
            "devices": [device] if device is not None else [],
            "prof_tools": status.get("prof_tools") or [],
            "test_examples": status.get("test_examples") or status.get("script_options") or [],
            "default_example_id": status.get("default_example_id"),
            "example_schema_version": status.get("example_schema_version"),
            "example_manifest_hash": status.get("example_manifest_hash"),
            "job_types": ["profile", "build_install"] if perf_config.mode == "ssh" else ["profile"],
            "op_warm_up": status.get("op_warm_up"),
            "op_launch_count": status.get("op_launch_count"),
            "max_concurrency": self.config.max_concurrency,
            "notification": self.notification.status(),
            "npu_status": npu_status,
            "execution_environment": {
                "defaults": execution_environment_defaults(perf_config),
                "customizable": perf_config.mode == "ssh",
                "source_build": perf_config.mode == "ssh" and bool(perf_config.remote_source_repo),
                "source_deployment": perf_config.mode == "ssh" and bool(perf_config.remote_source_repo),
                "source_branch_query": perf_config.mode == "ssh" and bool(perf_config.remote_source_repo),
                "source_remote_branch_query": perf_config.mode == "ssh" and bool(perf_config.remote_source_repo),
                "source_branches": self.source_branches_status(),
            },
            "agent_version": "2.0.1",
        }

    def source_branches_cache_path(self) -> Path:
        runner_key = hashlib.sha256(self.config.runner_id.encode("utf-8")).hexdigest()[:12]
        return self.config.state_dir / f"source-branches-{runner_key}.json"

    def _load_source_branches_cache(self) -> dict[str, Any]:
        empty = {
            "checked_at": None,
            "source_repo": None,
            "branches": [],
            "error": "等待源码分支查询",
            "refresh_request_id": None,
            "stale": False,
        }
        try:
            value = json.loads(self.source_branches_cache_path().read_text(encoding="utf-8"))
            source_repo = str(value.get("source_repo") or "").strip()
            branches = [
                {"source": str(branch["source"]), "name": str(branch["name"])}
                for branch in value.get("branches", [])[:400]
                if isinstance(branch, dict)
                and branch.get("source") in {"local", "remote"}
                and isinstance(branch.get("name"), str)
                and branch["name"]
            ]
            if not source_repo or not branches:
                return empty
            return {
                "checked_at": value.get("checked_at"),
                "source_repo": source_repo,
                "branches": branches,
                "error": "",
                "refresh_request_id": None,
                "stale": True,
            }
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return empty

    def _save_source_branches_cache(self, status: dict[str, Any]) -> None:
        if not status.get("branches") or not isinstance(getattr(self.config, "state_dir", None), Path):
            return
        path = self.source_branches_cache_path()
        temporary = path.with_suffix(".tmp")
        value = {
            "checked_at": status.get("checked_at"),
            "source_repo": status.get("source_repo"),
            "branches": status.get("branches"),
        }
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def source_branches_status(self) -> dict[str, Any]:
        with self._source_branches_lock:
            return {
                **self._source_branches,
                "branches": list(self._source_branches.get("branches") or []),
                "refreshing": self._source_branches_refreshing,
            }

    def request_source_branches_refresh(self, refresh_id: str, source_repo: str) -> None:
        refresh_id = str(refresh_id or "").strip()
        source_repo = str(source_repo or "").strip()
        if not refresh_id or not source_repo:
            return
        request = (refresh_id, source_repo)
        with self._source_branches_lock:
            if refresh_id == self._source_branches.get("refresh_request_id"):
                return
            if request == self._source_branches_pending:
                return
            if self._source_branches_refreshing:
                self._source_branches_pending = request
                return
            self._source_branches_refreshing = True
            threading.Thread(
                target=self._refresh_source_branches,
                args=request,
                name=f"source-branches-{self.config.runner_id}",
                daemon=True,
            ).start()

    def npu_status(self, *, refresh: bool = True) -> dict[str, Any]:
        now = time.monotonic()
        with self._npu_status_lock:
            interval = self.config.npu_status_interval_seconds
            stale = now - self._npu_status_checked_at >= interval
            if refresh and stale and not self._npu_status_refreshing:
                self._npu_status_refreshing = True
                self._npu_status_checked_at = now
                threading.Thread(
                    target=self._refresh_npu_status,
                    args=(None,),
                    name=f"npu-status-{self.config.runner_id}",
                    daemon=True,
                ).start()
            return {
                **self._npu_status,
                "devices": [dict(device) for device in self._npu_status.get("devices", [])],
            }

    def request_npu_status_refresh(self, refresh_id: str) -> None:
        refresh_id = str(refresh_id or "").strip()
        if not refresh_id:
            return
        with self._npu_status_lock:
            if refresh_id == self._npu_status.get("refresh_request_id"):
                return
            if refresh_id == self._npu_status_pending_refresh_id:
                return
            if self._npu_status_refreshing:
                self._npu_status_pending_refresh_id = refresh_id
                return
            self._npu_status_refreshing = True
            self._npu_status_checked_at = time.monotonic()
            threading.Thread(
                target=self._refresh_npu_status,
                args=(refresh_id,),
                name=f"npu-status-forced-{self.config.runner_id}",
                daemon=True,
            ).start()

    def _handle_runner_control(self, response: dict[str, Any]) -> None:
        runner = response.get("runner", {})
        refresh_id = runner.get("npu_status_refresh_id")
        if refresh_id:
            self.request_npu_status_refresh(str(refresh_id))
        source_branches_refresh_id = runner.get("source_branches_refresh_id")
        source_branches_repo = runner.get("source_branches_repo")
        if source_branches_refresh_id and source_branches_repo:
            self.request_source_branches_refresh(
                str(source_branches_refresh_id),
                str(source_branches_repo),
            )

    def _refresh_source_branches(self, refresh_id: str, source_repo: str) -> None:
        checked_at = utc_now()
        with self._source_branches_lock:
            cached_branches = (
                list(self._source_branches.get("branches") or [])
                if self._source_branches.get("source_repo") == source_repo
                else []
            )
        try:
            branches = list_remote_source_branches(load_config(), source_repo)
            status = {
                "checked_at": checked_at,
                "source_repo": source_repo,
                "branches": branches,
                "error": "",
                "refresh_request_id": refresh_id,
                "stale": False,
            }
            self._save_source_branches_cache(status)
        except Exception as exc:
            status = {
                "checked_at": checked_at,
                "source_repo": source_repo,
                "branches": cached_branches,
                "error": str(exc)[:500],
                "refresh_request_id": refresh_id,
                "stale": bool(cached_branches),
            }
        with self._source_branches_lock:
            self._source_branches = status
            self._source_branches_refreshing = False
            pending = self._source_branches_pending
            self._source_branches_pending = None
            if pending and pending[0] != refresh_id:
                self._source_branches_refreshing = True
                threading.Thread(
                    target=self._refresh_source_branches,
                    args=pending,
                    name=f"source-branches-{self.config.runner_id}",
                    daemon=True,
                ).start()
        try:
            self.send_runner_heartbeat(self.health())
        except Exception:
            pass

    def _refresh_npu_status(self, refresh_id: str | None) -> None:
        with self._npu_status_lock:
            completed_refresh_id = self._npu_status.get("refresh_request_id")
        try:
            perf_config = load_config()
            devices = collect_npu_device_status(
                perf_config,
                device_count=self.config.npu_device_count,
                timeout_seconds=self.config.npu_status_timeout_seconds,
            )
            available = sum(1 for device in devices if device.get("available"))
            checked_at = utc_now()
            status = {
                "updated_at": checked_at,
                "checked_at": checked_at,
                "devices": devices,
                "error": "" if available else "npu-smi 未返回可用设备",
                "refresh_request_id": refresh_id or completed_refresh_id,
            }
        except Exception as exc:
            with self._npu_status_lock:
                previous = self._npu_status
            status = {
                "updated_at": previous.get("updated_at"),
                "checked_at": utc_now(),
                "devices": previous.get("devices", []),
                "error": str(exc)[:500],
                "refresh_request_id": refresh_id or previous.get("refresh_request_id"),
            }
        with self._npu_status_lock:
            self._npu_status = status
            self._npu_status_checked_at = time.monotonic()
            self._npu_status_refreshing = False
            pending_refresh_id = self._npu_status_pending_refresh_id
            self._npu_status_pending_refresh_id = None
            if pending_refresh_id and pending_refresh_id != status.get("refresh_request_id"):
                self._npu_status_refreshing = True
                threading.Thread(
                    target=self._refresh_npu_status,
                    args=(pending_refresh_id,),
                    name=f"npu-status-forced-{self.config.runner_id}",
                    daemon=True,
                ).start()
        try:
            health = self.health()
            self.send_runner_heartbeat(health)
        except Exception:
            pass

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

    def runner_payload(self, health: dict[str, Any], current_jobs: int | None = None) -> dict[str, Any]:
        if current_jobs is None:
            current_jobs = self.active_job_count()
        return {
            "runner_id": self.config.runner_id,
            "name": self.config.runner_name,
            "capabilities": self.capabilities(refresh_npu=bool(health.get("npu_reachable"))),
            "vpn_connected": bool(health.get("vpn_connected")),
            "npu_reachable": bool(health.get("npu_reachable")),
            "current_jobs": current_jobs,
            "last_error": str(health.get("last_error") or "")[:1000],
        }

    def send_runner_heartbeat(self, health: dict[str, Any], current_jobs: int | None = None) -> dict[str, Any]:
        response = self.api.post("/api/runner/heartbeat", self.runner_payload(health, current_jobs))
        self._handle_runner_control(response)
        return response

    def register(self) -> dict[str, Any]:
        health = self.health()
        response = self.api.post("/api/runner/register", self.runner_payload(health))
        self._handle_runner_control(response)
        return response

    def claim(self, health: dict[str, Any]) -> dict[str, Any] | None:
        response = self.api.post("/api/runner/jobs/claim", self.runner_payload(health))
        self._handle_runner_control(response)
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

    def load_job_state(self, job_id: str) -> dict[str, Any]:
        try:
            return json.loads(self.job_state_path(job_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_job_state(self, job: dict[str, Any], state: str, **extra: Any) -> None:
        path = self.job_state_path(job["id"])
        previous = self.load_job_state(job["id"])
        record = {
            **previous,
            "job_id": job["id"],
            "attempt_id": job.get("attempt_id"),
            "runner_id": self.config.runner_id,
            "state": state,
            "updated_at": utc_now(),
            "request": dict(job.get("request") or previous.get("request") or {}),
            **extra,
        }
        if state in {"claimed", "running", "reporting", "reporting_failure", "reporting_orphaned"} and job.get("lease_token"):
            record["lease_token"] = job["lease_token"]
        elif state in {"completed", "failed", "canceled", "orphaned"}:
            record.pop("lease_token", None)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def recoverable_build_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted((self.config.state_dir / "jobs").glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            request = dict(record.get("request") or {})
            if (
                record.get("state") not in {"claimed", "running", "reporting", "reporting_failure", "reporting_orphaned"}
                or request.get("task_type") != "build_install"
                or not record.get("job_id")
                or not record.get("attempt_id")
                or not record.get("lease_token")
            ):
                continue
            job = {
                "id": record["job_id"],
                "attempt_id": record["attempt_id"],
                "lease_token": record["lease_token"],
                "request": request,
                "remote_build": record.get("remote_build") or {},
                "recovery_state": record,
            }
            jobs.append(job)
        return jobs

    def recover_build_jobs(self) -> int:
        recovered = 0
        for job in self.recoverable_build_jobs():
            try:
                self.run_job(job, resume=True)
                recovered += 1
            except Exception as exc:
                print(f"[runner] 恢复编译任务 {job['id']} 失败：{exc}", flush=True)
        return recovered

    def schedule_recovered_build_jobs(self) -> int:
        scheduled = 0
        for job in self.recoverable_build_jobs():
            if not self.submit_job(job, resume=True):
                break
            scheduled += 1
        return scheduled

    def run_persistent_build_job(
        self,
        job: dict[str, Any],
        heartbeat: JobHeartbeat,
        *,
        resume: bool,
    ) -> dict[str, Any]:
        request = dict(job.get("request") or {})
        config = load_config()
        handle = dict(job.get("remote_build") or {})
        if not handle:
            handle = persistent_build_handle(request, str(job["attempt_id"]), config)
        self.save_job_state(job, "running", request=request, remote_build=handle)

        def poll_when_reachable(*, allow_missing: bool = False) -> dict[str, Any]:
            while True:
                try:
                    return poll_persistent_build_install(handle, config, allow_missing=allow_missing)
                except RemoteConnectionError as exc:
                    self.save_job_state(
                        job,
                        "running",
                        request=request,
                        remote_build=handle,
                        connection_error=str(exc),
                    )
                    time.sleep(5)

        status = poll_when_reachable(allow_missing=True)
        heartbeat.set_remote_state(str(status.get("state") or "unknown"))
        if status.get("state") == "missing":
            while True:
                try:
                    handle = start_persistent_build_install(request, handle, config)
                    break
                except RemoteConnectionError as exc:
                    self.save_job_state(
                        job,
                        "running",
                        request=request,
                        remote_build=handle,
                        connection_error=str(exc),
                    )
                    time.sleep(5)
            job["remote_build"] = handle
            self.save_job_state(job, "running", request=request, remote_build=handle)
        elif not resume and not status.get("active") and status.get("state") != "succeeded":
            raise RuntimeError(f"远端编译任务处于不可启动状态：{status.get('state')}")

        last_log_size = -1
        last_event_at = 0.0
        cancel_sent = False
        started_waiting_at = time.monotonic()
        while True:
            if heartbeat.cancel_requested.is_set() and not cancel_sent:
                cancel_persistent_build_install(handle, config)
                cancel_sent = True
            status = poll_when_reachable()
            heartbeat.set_remote_state(str(status.get("state") or "unknown"))
            if status.get("commit"):
                handle["commit"] = status["commit"]
            self.save_job_state(
                job,
                "running",
                request=request,
                remote_build=handle,
                remote_status={
                    "state": status.get("state"),
                    "pid": status.get("pid"),
                    "alive": status.get("alive"),
                    "exit_code": status.get("exit_code"),
                    "log_size": status.get("log_size"),
                    "log_tail": str(status.get("log_tail") or "")[-16000:],
                },
            )
            now = time.monotonic()
            log_size = int(status.get("log_size") or 0)
            if log_size != last_log_size and now - last_event_at >= 30:
                lines = [line.strip() for line in str(status.get("log_tail") or "").splitlines() if line.strip()]
                message = lines[-1] if lines else "远端编译正在执行"
                try:
                    self.api.post(
                        f"/api/runner/jobs/{job['id']}/events",
                        self.job_auth(job, {
                            "event_type": "build_progress",
                            "level": "info",
                            "message": message[-1000:],
                            "detail": {
                                "remote_log": handle.get("log_path"),
                                "log_size": log_size,
                                "state": status.get("state"),
                                "log_tail": str(status.get("log_tail") or "")[-8000:],
                            },
                        }),
                    )
                except Exception:
                    pass
                last_log_size = log_size
                last_event_at = now
            if status.get("state") == "missing" and now - started_waiting_at < 30:
                time.sleep(2)
                continue
            if not status.get("active"):
                return persistent_build_result(handle, status)
            time.sleep(5)

    def report_persisted_outcome(self, job: dict[str, Any], record: dict[str, Any]) -> bool:
        state = str(record.get("state") or "")
        request = dict(job.get("request") or record.get("request") or {})
        if state == "reporting":
            payload = record.get("completion_payload")
            if not isinstance(payload, dict):
                raise RuntimeError("本地编译成功记录缺少 completion_payload")
            try:
                self.api.post(f"/api/runner/jobs/{job['id']}/complete", payload)
            except Exception as exc:
                print(f"[runner] 编译任务 {job['id']} 成功结果回传失败，将继续重试：{exc}", flush=True)
                return False
            remote_build = dict(record.get("remote_build") or self.load_job_state(job["id"]).get("remote_build") or {})
            if request.get("task_type") == "build_install" and remote_build:
                try:
                    acknowledge_persistent_build_install(remote_build, load_config())
                except Exception as exc:
                    print(f"[runner] 编译任务 {job['id']} 远端清理延后：{exc}", flush=True)
            self.save_job_state(
                job,
                "completed",
                request=request,
                artifacts=record.get("artifacts") or [],
                message=record.get("message") or "",
                upload_errors=record.get("upload_errors") or [],
            )
            return True
        if state == "reporting_failure":
            payload = record.get("failure_payload")
            if not isinstance(payload, dict):
                raise RuntimeError("本地编译失败记录缺少 failure_payload")
            try:
                self.api.post(f"/api/runner/jobs/{job['id']}/fail", payload)
            except Exception as exc:
                print(f"[runner] 编译任务 {job['id']} 失败结果回传失败，将继续重试：{exc}", flush=True)
                return False
            terminal_state = "canceled" if payload.get("canceled") else "failed"
            self.save_job_state(job, terminal_state, request=request, error=record.get("error") or "")
            return True
        if state == "reporting_orphaned":
            payload = record.get("reconcile_payload")
            if not isinstance(payload, dict):
                raise RuntimeError("本地编译状态确认记录缺少 reconcile_payload")
            try:
                self.api.post(f"/api/runner/jobs/{job['id']}/reconcile", payload)
            except Exception as exc:
                print(f"[runner] 编译任务 {job['id']} 状态确认回传失败，将继续重试：{exc}", flush=True)
                return False
            self.save_job_state(job, "orphaned", request=request, error=record.get("error") or "")
            return True
        return False

    def finish_recovered_final_job(self, job: dict[str, Any], response: dict[str, Any]) -> bool:
        final_status = str(response.get("final_status") or (response.get("job") or {}).get("status") or "")
        local_state = {
            "succeeded": "completed",
            "failed": "failed",
            "canceled": "canceled",
            "orphaned": "orphaned",
        }.get(final_status)
        if not local_state:
            return False
        self.save_job_state(
            job,
            local_state,
            request=dict(job.get("request") or {}),
            message=f"Worker 已将任务标记为 {final_status}，停止恢复旧尝试",
        )
        return True

    def run_job(self, job: dict[str, Any], *, resume: bool = False) -> None:
        job_id = str(job["id"])
        self._mark_job_active(job_id)
        try:
            self._run_job(job, resume=resume)
        finally:
            self._mark_job_inactive(job_id)
            try:
                self.send_runner_heartbeat(self.health())
            except Exception:
                pass

    def _run_job(self, job: dict[str, Any], *, resume: bool = False) -> None:
        request = dict(job.get("request") or {})
        task_type = str(request.get("task_type") or "profile")
        task_label = "编译安装" if task_type == "build_install" else "测试"
        recovery_state = dict(job.get("recovery_state") or {})
        if not resume:
            self.save_job_state(job, "claimed", request=request)
        if resume and recovery_state.get("state") in {"claimed", "running"}:
            try:
                response = self.api.post(
                    f"/api/runner/jobs/{job['id']}/heartbeat",
                    self.job_auth(job, {
                        "state": "running",
                        "remote_state": (recovery_state.get("remote_status") or {}).get("state") or "recovering",
                        "message": "Relay 重启后正在确认原编译尝试",
                    }),
                )
            except Exception as exc:
                print(f"[runner] 任务 {job['id']} 恢复握手失败，未启动远端命令：{exc}", flush=True)
                return
            if response.get("final") and self.finish_recovered_final_job(job, response):
                return
        if not resume or recovery_state.get("state") == "claimed":
            try:
                response = self.api.post(
                    f"/api/runner/jobs/{job['id']}/started",
                    self.job_auth(job, {
                        "remote_execution_id": f"{self.config.runner_id}:{job['id']}:{job.get('attempt_id')}",
                        "message": f"Relay 已开始执行{task_label}任务",
                    }),
                )
            except Exception as exc:
                print(f"[runner] 任务 {job['id']} 启动确认失败，将继续重试：{exc}", flush=True)
                return
            if response.get("final") and self.finish_recovered_final_job(job, response):
                return
        heartbeat = JobHeartbeat(self, job)
        heartbeat.start()
        try:
            if resume and recovery_state.get("state") in {"reporting", "reporting_failure", "reporting_orphaned"}:
                self.report_persisted_outcome(job, recovery_state)
                return
            self.save_job_state(job, "running", request=request)
            try:
                if task_type == "build_install":
                    build_slot = getattr(self, "_build_slot", None) or nullcontext()
                    with build_slot:
                        result = self.run_persistent_build_job(job, heartbeat, resume=resume)
                else:
                    run_id = f"{job['id']}-{job.get('attempt_id') or 'attempt'}"
                    result = execute(request, persist_local_data=False, run_id=run_id)
                artifacts, local_artifacts = self.build_artifacts(job, result)
                upload_errors: list[str] = []
                if artifacts:
                    upload_slot = getattr(self, "_upload_slots", None) or nullcontext()
                    with upload_slot:
                        artifacts, local_artifacts, upload_errors = self.upload_artifacts(
                            job,
                            artifacts,
                            local_artifacts,
                        )
                if upload_errors:
                    try:
                        self.api.post(
                            f"/api/runner/jobs/{job['id']}/events",
                            self.job_auth(job, {
                                "event_type": "artifact_upload_failed",
                                "level": "warning",
                                "message": "R2 上传失败，原始制品仅保留在 Relay 本地",
                                "detail": {"errors": upload_errors},
                            }),
                        )
                    except Exception:
                        pass
                if heartbeat.cancel_requested.is_set():
                    raise InterruptedError(f"{task_label}完成前收到取消请求")
                snapshot = result.get("snapshot") or {}
                completion_payload = self.job_auth(job, {
                    "exit_code": 0,
                    "task_type": task_type,
                    "message": result.get("message") or f"{task_label}完成",
                    "command": result.get("command") or "",
                    "profiler_command": result.get("profiler_command") or "",
                    "environment": {
                        **self.environment_summary(),
                        **(result.get("execution_environment") or {}),
                    },
                    "metrics": extract_snapshot_metrics(snapshot),
                    "snapshot": snapshot,
                    "perf_data": result.get("data") or {},
                    "result": {
                        "task_type": task_type,
                        "message": result.get("message") or "",
                        "prof_tool": result.get("prof_tool") or request.get("prof_tool"),
                        "prof_source": result.get("prof_source") or snapshot.get("prof_source"),
                        "execution_environment": result.get("execution_environment") or {},
                    },
                    "artifacts": artifacts,
                })
                outcome_record = {
                    "state": "reporting",
                    "request": request,
                    "remote_build": job.get("remote_build") or {},
                    "artifacts": local_artifacts,
                    "message": result.get("message", ""),
                    "upload_errors": upload_errors,
                    "completion_payload": completion_payload,
                }
                self.save_job_state(
                    job,
                    "reporting",
                    **{key: value for key, value in outcome_record.items() if key != "state"},
                )
                self.report_persisted_outcome(job, outcome_record)
            except Exception as exc:
                if isinstance(exc, RemoteBuildOrphanedError):
                    reconcile_payload = self.job_auth(job, {
                        "remote_state": "missing",
                        "message": str(exc),
                        "detail": {
                            "error_type": exc.__class__.__name__,
                            "remote_build": job.get("remote_build") or {},
                        },
                    })
                    outcome_record = {
                        "state": "reporting_orphaned",
                        "request": request,
                        "error": str(exc),
                        "reconcile_payload": reconcile_payload,
                    }
                    self.save_job_state(
                        job,
                        "reporting_orphaned",
                        **{key: value for key, value in outcome_record.items() if key != "state"},
                    )
                    self.report_persisted_outcome(job, outcome_record)
                    return
                canceled = heartbeat.cancel_requested.is_set() or isinstance(exc, InterruptedError)
                failure_payload = self.job_auth(job, {
                    "message": str(exc),
                    "error_type": exc.__class__.__name__,
                    "canceled": canceled,
                })
                outcome_record = {
                    "state": "reporting_failure",
                    "request": request,
                    "error": str(exc),
                    "failure_payload": failure_payload,
                }
                self.save_job_state(
                    job,
                    "reporting_failure",
                    **{key: value for key, value in outcome_record.items() if key != "state"},
                )
                self.report_persisted_outcome(job, outcome_record)
        finally:
            heartbeat.stop()
            heartbeat.join(timeout=2)

    def environment_summary(self) -> dict[str, Any]:
        status = runner_status()
        return {
            "agent_version": "1.9.0",
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

    def upload_artifacts(
        self,
        job: dict[str, Any],
        artifacts: list[dict[str, Any]],
        local_artifacts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        uploaded: list[dict[str, Any]] = []
        errors: list[str] = []
        for artifact, local in zip(artifacts, local_artifacts):
            archive_path: Path | None = None
            try:
                source = Path(local["local_path"]).resolve()
                archive_path = self.create_artifact_archive(job["id"], source)
                archive_size, archive_sha256 = file_digest(archive_path)
                cloud_artifact = {
                    "id": f"artifact-{archive_sha256[:32]}",
                    "type": "prof_archive",
                    "filename": f"{source.name}.zip",
                    "content_type": "application/zip",
                    "size_bytes": archive_size,
                    "sha256": archive_sha256,
                    "expires_at": artifact["expires_at"],
                }
                stored = self.multipart_upload(job, archive_path, cloud_artifact)
                uploaded.append(stored)
                local.update({
                    "storage": "r2",
                    "cloud_artifact": stored,
                    "source_size_bytes": local.get("size_bytes"),
                    "source_sha256": local.get("sha256"),
                })
            except Exception as exc:
                message = f"{artifact.get('filename') or 'artifact'}: {exc}"
                errors.append(message)
                local["upload_error"] = str(exc)
                uploaded.append(artifact)
            finally:
                if archive_path and archive_path.exists():
                    archive_path.unlink()
        return uploaded, local_artifacts, errors

    def create_artifact_archive(self, job_id: str, source: Path) -> Path:
        upload_dir = self.config.state_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        archive_path = upload_dir / f"{job_id}-{source.name}.zip"
        temporary = archive_path.with_suffix(".zip.tmp")
        temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            if source.is_file():
                archive.write(source, arcname=source.name)
            else:
                for item in sorted(path for path in source.rglob("*") if path.is_file()):
                    archive.write(item, arcname=(Path(source.name) / item.relative_to(source)).as_posix())
        temporary.replace(archive_path)
        return archive_path

    def multipart_upload(
        self,
        job: dict[str, Any],
        archive_path: Path,
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        start = self.api.post(
            f"/api/runner/jobs/{job['id']}/artifacts/multipart/start",
            self.job_auth(job, {"artifact": artifact}),
        )
        upload_id = str(start["upload_id"])
        artifact_id = str(start["artifact_id"])
        parts: list[dict[str, Any]] = []
        query = urllib.parse.urlencode({"upload_id": upload_id})
        headers = {
            "X-Runner-Id": self.config.runner_id,
            "X-Attempt-Id": str(job["attempt_id"]),
            "X-Lease-Token": str(job["lease_token"]),
        }
        try:
            with archive_path.open("rb") as handle:
                part_number = 1
                while chunk := handle.read(self.config.upload_part_bytes):
                    response = self.api.put(
                        f"/api/runner/jobs/{job['id']}/artifacts/multipart/"
                        f"{artifact_id}/parts/{part_number}?{query}",
                        chunk,
                        headers,
                    )
                    parts.append(response["part"])
                    part_number += 1
            completed = self.api.post(
                f"/api/runner/jobs/{job['id']}/artifacts/multipart/{artifact_id}/complete",
                self.job_auth(job, {
                    "upload_id": upload_id,
                    "parts": parts,
                    "artifact": artifact,
                }),
            )
            return completed["artifact"]
        except Exception:
            try:
                self.api.post(
                    f"/api/runner/jobs/{job['id']}/artifacts/multipart/{artifact_id}/abort",
                    self.job_auth(job, {"upload_id": upload_id}),
                )
            except Exception:
                pass
            raise

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
        if not health["vpn_connected"] or not health["npu_reachable"]:
            self.send_runner_heartbeat(health)
            return False
        job = self.claim(health)
        if not job:
            return False
        self.run_job(job)
        return True

    def run_forever(self) -> None:
        self.register()
        self.schedule_recovered_build_jobs()
        notification_active = self.notification.start()
        delay = self.config.poll_min_seconds
        next_reconcile_at = time.monotonic() + self.config.reconcile_seconds
        last_runner_heartbeat_at = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    worked = False
                    self.cleanup_expired_artifacts()
                    health = self.health()
                    if not health["vpn_connected"] or not health["npu_reachable"]:
                        self.send_runner_heartbeat(health)
                        last_runner_heartbeat_at = time.monotonic()
                    while (
                        health["vpn_connected"]
                        and health["npu_reachable"]
                        and self.available_slots() > 0
                        and not self.stop_event.is_set()
                    ):
                        job = self.claim(health)
                        if not job:
                            break
                        if not self.submit_job(job):
                            break
                        worked = True
                    now = time.monotonic()
                    if now - last_runner_heartbeat_at >= self.config.heartbeat_seconds:
                        self.send_runner_heartbeat(health)
                        last_runner_heartbeat_at = time.monotonic()
                    heartbeat_delay = max(
                        0.1,
                        self.config.heartbeat_seconds - (time.monotonic() - last_runner_heartbeat_at),
                    )
                    if notification_active:
                        delay = min(max(0.1, next_reconcile_at - time.monotonic()), heartbeat_delay)
                    else:
                        poll_delay = self.config.poll_min_seconds if worked else min(self.config.poll_max_seconds, delay + 2)
                        delay = min(poll_delay, heartbeat_delay)
                except Exception as exc:
                    print(f"[runner] {exc}", flush=True)
                    delay = min(self.config.error_backoff_max_seconds, max(delay * 2, 5))
                event_received = self._dispatch_event.wait(delay)
                self._dispatch_event.clear()
                if notification_active and not event_received:
                    self.notification.mark_reconcile()
                    next_reconcile_at = time.monotonic() + self.config.reconcile_seconds
        finally:
            self.notification.stop()
            self._executor.shutdown(wait=True, cancel_futures=False)


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


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
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
