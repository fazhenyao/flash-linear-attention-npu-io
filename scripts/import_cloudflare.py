#!/usr/bin/env python3
"""Import repository snapshots into the Cloudflare Worker D1 backend."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def post_json(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/import",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"导入失败：HTTP {exc.code} {body}") from exc


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="把当前仓库快照导入 Cloudflare D1 后端。")
    parser.add_argument("--api", required=True, help="Worker URL，例如 https://xxx.workers.dev")
    parser.add_argument("--token", required=True, help="Wrangler secret ADMIN_TOKEN 的值")
    parser.add_argument("--state", default=str(ROOT / "data" / "project-state.json"))
    parser.add_argument("--audit", default=str(ROOT / "data" / "audit-log.jsonl"))
    parser.add_argument("--pr-catalog", default=str(ROOT / "data" / "pr-catalog.json"))
    parser.add_argument("--perf", default=str(ROOT / "data" / "performance-data.json"))
    args = parser.parse_args()

    state = read_json(Path(args.state))
    if not state:
        print("未找到项目状态快照。", file=sys.stderr)
        return 1
    payload = {
        "state": state,
        "audit": read_jsonl(Path(args.audit)),
        "prCatalog": read_json(Path(args.pr_catalog), {"generatedAt": "", "sourceRepo": "", "total": 0, "items": []}),
        "perfData": read_json(Path(args.perf), {"version": "", "models": [], "cases": [], "snapshots": [], "runs": []}),
    }
    result = post_json(args.api, args.token, payload)
    task_count = len(result.get("state", {}).get("tasks", []))
    print(f"导入完成：{task_count} 项任务")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
