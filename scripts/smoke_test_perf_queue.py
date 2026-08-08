#!/usr/bin/env python3
"""Exercise the Cloudflare performance queue API without running NPU commands."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def request_json(api: str, path: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api.rstrip("/") + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "fla-perf-queue-smoke-test",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {exc.code} {body}") from exc


def expect_http_error(api: str, path: str, token: str, status: int, payload: dict | None = None) -> None:
    try:
        request_json(api, path, token, "POST", payload)
    except RuntimeError as exc:
        if f"HTTP {status}" in str(exc):
            return
        raise
    raise RuntimeError(f"POST {path}: expected HTTP {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--admin-token", required=True)
    parser.add_argument("--runner-token", required=True)
    args = parser.parse_args()

    suffix = uuid.uuid4().hex
    runner_id = f"smoke-runner-{suffix[:10]}"
    runner_payload = {
        "runner_id": runner_id,
        "name": "Smoke Test Runner",
        "capabilities": {
            "mode": "test",
            "chips": ["A2"],
            "devices": [2],
            "prof_tools": ["msprof", "msprof_op"],
            "max_concurrency": 1,
        },
        "vpn_connected": True,
        "npu_reachable": True,
        "current_jobs": 0,
    }
    request_json(args.api, "/api/runner/register", args.runner_token, "POST", runner_payload)
    job_payload = {
        "idempotency_key": f"smoke-{suffix}",
        "prof_tool": "msprof",
        "script_id": "scripts/flash_gated_delta_rule.py",
        "model_id": "gdn",
        "chip": "A2",
        "device": 2,
        "attributes": {"tokens": 128, "query_heads": 4, "value_heads": 4},
    }
    created = request_json(args.api, "/api/perf/jobs", args.admin_token, "POST", job_payload)
    job_id = created["job"]["id"]
    duplicate = request_json(args.api, "/api/perf/jobs", args.admin_token, "POST", job_payload)
    if not duplicate.get("duplicate") or duplicate["job"]["id"] != job_id:
        raise RuntimeError("idempotency check failed")
    expect_http_error(args.api, "/api/perf/jobs", args.admin_token, 400, {
        **job_payload,
        "idempotency_key": f"invalid-{suffix}",
        "command": "echo should-not-run",
    })
    claimed = request_json(args.api, "/api/runner/jobs/claim", args.runner_token, "POST", runner_payload)
    job = claimed.get("job")
    if not job or job["id"] != job_id:
        raise RuntimeError(f"expected {job_id}, claimed {job and job.get('id')}")
    lease = {
        "runner_id": runner_id,
        "attempt_id": job["attempt_id"],
        "lease_token": job["lease_token"],
    }
    request_json(args.api, f"/api/runner/jobs/{job_id}/started", args.runner_token, "POST", {
        **lease,
        "remote_execution_id": f"smoke:{job_id}",
    })
    request_json(args.api, f"/api/runner/jobs/{job_id}/events", args.runner_token, "POST", {
        **lease,
        "event_type": "smoke",
        "message": "smoke test event",
    })
    request_json(args.api, f"/api/runner/jobs/{job_id}/complete", args.runner_token, "POST", {
        **lease,
        "exit_code": 0,
        "message": "smoke test complete",
        "environment": {"agent_version": "smoke"},
        "metrics": {"duration_us": 1.25},
        "result": {"kind": "smoke"},
        "artifacts": [{
            "type": "test",
            "object_key": f"relay://{runner_id}/{job_id}/result.json",
            "filename": "result.json",
            "content_type": "application/json",
            "size_bytes": 2,
            "sha256": "0" * 64,
        }],
    })
    result = request_json(args.api, f"/api/perf/jobs/{job_id}", args.admin_token)
    if result["job"]["status"] != "succeeded":
        raise RuntimeError(f"unexpected final state: {result['job']['status']}")

    cancel_payload = {**job_payload, "idempotency_key": f"cancel-{suffix}"}
    cancel_job = request_json(args.api, "/api/perf/jobs", args.admin_token, "POST", cancel_payload)["job"]
    canceled = request_json(
        args.api, f"/api/perf/jobs/{cancel_job['id']}/cancel", args.admin_token, "POST", {},
    )["job"]
    if canceled["status"] != "canceled":
        raise RuntimeError(f"cancel failed: {canceled['status']}")
    retried = request_json(
        args.api, f"/api/perf/jobs/{cancel_job['id']}/retry", args.admin_token, "POST", {},
    )["job"]
    if retried["status"] != "queued" or retried["retry_count"] != 1:
        raise RuntimeError(f"retry failed: {retried['status']} / {retried['retry_count']}")
    request_json(args.api, f"/api/perf/jobs/{cancel_job['id']}/cancel", args.admin_token, "POST", {})
    print(json.dumps({
        "ok": True,
        "job_id": job_id,
        "status": result["job"]["status"],
        "runner_id": runner_id,
        "idempotency": True,
        "validation": True,
        "cancel_retry": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
