#!/usr/bin/env python3
"""Back up Cloudflare D1-backed project data into repository snapshots."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_PATHS = [ROOT / "data" / "project-state.json", ROOT / "docs" / "project-state.json"]
AUDIT_PATHS = [ROOT / "data" / "audit-log.jsonl", ROOT / "docs" / "audit-log.jsonl"]
CATALOG_PATHS = [ROOT / "data" / "pr-catalog.json", ROOT / "docs" / "pr-catalog.json"]
PERF_PATHS = [ROOT / "data" / "performance-data.json", ROOT / "docs" / "performance-data.json"]


def default_worker_api() -> str:
  config_path = ROOT / "docs" / "config.js"
  if not config_path.exists():
    return ""
  match = re.search(
    r"FLASH_IO_API_BASE\s*=\s*[\"']([^\"']+)[\"']",
    config_path.read_text(encoding="utf-8-sig"),
  )
  return match.group(1).rstrip("/") if match else ""


def get_json(api: str, path: str, token: str | None = None) -> Any:
  headers = {
    "Accept": "application/json",
    "User-Agent": "flash-linear-attention-npu-io-backup",
  }
  if token:
    headers["Authorization"] = f"Bearer {token}"
  request = urllib.request.Request(api.rstrip("/") + path, headers=headers)
  try:
    with urllib.request.urlopen(request, timeout=60) as response:
      return json.loads(response.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"GET {path} failed: HTTP {exc.code} {body}") from exc


def write_json(paths: list[Path], payload: Any) -> None:
  text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(paths: list[Path], entries: list[dict[str, Any]]) -> None:
  text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
  parser = argparse.ArgumentParser(description="Back up Cloudflare D1 data into repository snapshot files.")
  parser.add_argument("--api", default=os.environ.get("FLASH_IO_API_BASE") or default_worker_api())
  parser.add_argument("--token", default=os.environ.get("FLASH_IO_ADMIN_TOKEN") or os.environ.get("ADMIN_TOKEN"))
  args = parser.parse_args()

  if not args.api:
    print("Worker API URL is not configured.", file=sys.stderr)
    return 1
  if not args.token:
    print("FLASH_IO_ADMIN_TOKEN is required for full audit backup.", file=sys.stderr)
    return 1

  state = get_json(args.api, "/api/export")
  audit = get_json(args.api, "/api/audit/export", args.token)
  catalog = get_json(args.api, "/api/pr-catalog")
  perf = get_json(args.api, "/api/perf")

  if not isinstance(state, dict) or not isinstance(state.get("tasks"), list):
    raise RuntimeError("Invalid state payload from /api/export.")
  if not isinstance(audit, list):
    raise RuntimeError("Invalid audit payload from /api/audit/export.")
  if not isinstance(catalog, dict) or not isinstance(catalog.get("items"), list):
    raise RuntimeError("Invalid catalog payload from /api/pr-catalog.")
  if not isinstance(perf, dict) or not isinstance(perf.get("snapshots"), list):
    raise RuntimeError("Invalid performance payload from /api/perf.")

  write_json(STATE_PATHS, state)
  write_jsonl(AUDIT_PATHS, audit)
  write_json(CATALOG_PATHS, catalog)
  write_json(PERF_PATHS, perf)

  print(json.dumps({
    "tasks": len(state.get("tasks", [])),
    "auditEntries": len(audit),
    "prCatalogItems": len(catalog.get("items", [])),
    "performanceSnapshots": len(perf.get("snapshots", [])),
  }, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
