#!/usr/bin/env python3
"""Run a real msprof performance test from trigger payload JSON or CLI flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from perf_runner import build_command, execute, runner_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a real GDN msprof performance test")
    parser.add_argument("--payload", type=Path, help="JSON payload file from performance dashboard trigger")
    parser.add_argument("--chip", default="A2", choices=["A2", "A3"])
    parser.add_argument("--model", default="gdn")
    parser.add_argument("--status", action="store_true", help="Print runner configuration and exit")
    parser.add_argument("--dry-run", action="store_true", help="Only print the command")
    return parser.parse_args()


def default_payload(chip: str, model: str) -> dict:
    return {
        "case_id": "manual",
        "model_id": model,
        "chip": chip,
        "device": "722",
        "attributes": {
            "batch": 1,
            "query_heads": 32,
            "value_heads": 32,
            "tokens": 4087,
            "key_dim": 128,
            "value_dim": 128,
            "chunk_size": 64,
            "dtype": "bf16",
            "mean_len": 1024,
            "cu_seqlens": "",
            "varlen": True,
        },
    }


def main() -> int:
    args = parse_args()
    if args.status:
        print(json.dumps(runner_status(), ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        import os

        os.environ["PERF_RUN_DRY_RUN"] = "1"
    payload = default_payload(args.chip, args.model)
    if args.payload:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    print("Command:")
    print(build_command(payload))
    result = execute(payload)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
