from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROF_APP_ROOT = ROOT / "data" / "prof_gdr"
PROF_OP_ROOT = ROOT / "data" / "prof_op"
VALID_PROF_TOOLS = {"msprof", "msprof_op", "msprof_op_sim"}

TRIGGER_SCRIPTS = [
    {
        "id": "flash-linear-attention-npu/examples/flash_gated_delta_rule.py",
        "label": "flash-linear-attention-npu/examples/flash_gated_delta_rule.py",
        "remote": "examples/flash_gated_delta_rule.py",
        "local": "ref/flash_gated_delta_rule.py",
    },
]


@dataclass
class PerfRunnerConfig:
    mode: str
    ssh_host: str
    ssh_user: str
    ssh_port: str
    ssh_identity: str
    remote_workdir: str
    remote_script: str
    local_script: Path
    npu_device: int
    chip: str
    prof_output_app: str
    prof_output_op: str
    soc_version: str
    dry_run: bool


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def load_config() -> PerfRunnerConfig:
    mode = os.environ.get("PERF_RUN_MODE", "auto").strip().lower()
    if mode == "mock":
        raise ValueError("已禁用模拟执行，请配置 PERF_RUN_MODE=ssh 或 local")
    if mode == "auto":
        if _env_bool("PERF_LOCAL_MODE") or os.environ.get("PERF_LOCAL_MODE", "").lower() == "local":
            mode = "local"
        elif os.environ.get("PERF_SSH_HOST", "").strip():
            mode = "ssh"
        else:
            mode = "unset"
    return PerfRunnerConfig(
        mode=mode,
        ssh_host=os.environ.get("PERF_SSH_HOST", "").strip(),
        ssh_user=os.environ.get("PERF_SSH_USER", "").strip() or "root",
        ssh_port=os.environ.get("PERF_SSH_PORT", "").strip(),
        ssh_identity=os.environ.get("PERF_SSH_IDENTITY_FILE", "").strip(),
        remote_workdir=os.environ.get("PERF_REMOTE_WORKDIR", "").strip() or "/data/flash-linear-attention-npu",
        remote_script=os.environ.get("PERF_REMOTE_SCRIPT", "").strip() or "examples/flash_gated_delta_rule.py",
        local_script=Path(os.environ.get("PERF_LOCAL_SCRIPT", "ref/flash_gated_delta_rule.py")),
        npu_device=int(os.environ.get("PERF_NPU_DEVICE", "2")),
        chip=os.environ.get("PERF_CHIP", "").strip().upper() or "A2",
        prof_output_app=os.environ.get("PERF_PROF_OUTPUT", "./prof_gdr").strip() or "./prof_gdr",
        prof_output_op=os.environ.get("PERF_OP_OUTPUT", "./prof_op").strip() or "./prof_op",
        soc_version=os.environ.get("PERF_SOC_VERSION", "").strip() or "Ascend910B",
        dry_run=_env_bool("PERF_RUN_DRY_RUN"),
    )


def resolve_script_paths(payload: dict[str, Any], config: PerfRunnerConfig) -> tuple[str, Path]:
    script_path = str(payload.get("script_path") or "").strip() or TRIGGER_SCRIPTS[0]["id"]
    entry = next(
        (item for item in TRIGGER_SCRIPTS if item["id"] == script_path or item["label"] == script_path),
        None,
    )
    if entry is None:
        allowed = ", ".join(item["label"] for item in TRIGGER_SCRIPTS)
        raise ValueError(f"未知脚本路径：{script_path}（可选：{allowed}）")
    local = Path(entry["local"])
    if not local.is_absolute():
        local = ROOT / local
    return entry["remote"], local


def script_options() -> list[dict[str, str]]:
    return [{"id": item["id"], "label": item["label"]} for item in TRIGGER_SCRIPTS]


def normalize_prof_tool(payload: dict[str, Any]) -> str:
    prof_tool = str(payload.get("prof_tool") or "msprof").strip()
    if prof_tool not in VALID_PROF_TOOLS:
        raise ValueError(f"prof_tool must be one of {sorted(VALID_PROF_TOOLS)}")
    return prof_tool


def prof_output_root(prof_tool: str, *, local: bool) -> Path:
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        return PROF_OP_ROOT if local else Path(load_config().prof_output_op)
    return PROF_APP_ROOT if local else Path(load_config().prof_output_app)


def prof_dir_prefix(prof_tool: str) -> str:
    return "OPPROF_" if prof_tool in {"msprof_op", "msprof_op_sim"} else "PROF_"


def prof_tool_label(prof_tool: str) -> str:
    return {
        "msprof": "msprof（整网）",
        "msprof_op": "msprof op（算子）",
        "msprof_op_sim": "msprof op simulator（仿真）",
    }.get(prof_tool, prof_tool)


def resolve_npu_device(payload: dict[str, Any], config: PerfRunnerConfig | None = None) -> int:
    raw = payload.get("device")
    if raw is not None and str(raw).strip() != "":
        device = int(raw)
        if device < 0:
            raise ValueError("device must be a non-negative integer")
        return device
    config = config or load_config()
    return config.npu_device


def resolve_chip(payload: dict[str, Any], config: PerfRunnerConfig | None = None) -> str:
    raw = str(payload.get("chip") or "").strip().upper()
    if raw:
        if raw not in {"A2", "A3"}:
            raise ValueError("chip must be A2 or A3")
        return raw
    config = config or load_config()
    chip = str(config.chip or "A2").strip().upper()
    if chip not in {"A2", "A3"}:
        raise ValueError("PERF_CHIP must be A2 or A3")
    return chip


def ensure_runner_configured() -> PerfRunnerConfig:
    config = load_config()
    if config.mode == "unset":
        raise ValueError(
            "未配置真实执行环境。请设置 PERF_RUN_MODE=ssh 与 PERF_SSH_HOST，"
            "或 PERF_RUN_MODE=local。参考 data/perf-runner.example.env"
        )
    if config.mode == "ssh" and not config.ssh_host:
        raise ValueError("PERF_SSH_HOST 未配置")
    if config.mode not in {"ssh", "local"}:
        raise ValueError(f"不支持的 PERF_RUN_MODE：{config.mode}")
    return config


def is_real_enabled() -> bool:
    try:
        ensure_runner_configured()
        return True
    except ValueError:
        return False


def runner_status() -> dict[str, Any]:
    error = None
    try:
        config = ensure_runner_configured()
        enabled = True
    except ValueError as exc:
        config = load_config()
        enabled = False
        error = str(exc)
    attrs = {
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
        "layout": "TND",
        "varlen": True,
    }
    payload = {"attributes": attrs, "prof_tool": "msprof"}
    payload_op = {**payload, "prof_tool": "msprof_op", "kernel_name": "chunk_bwd_dqkwg"}
    return {
        "enabled": enabled,
        "mode": config.mode,
        "dry_run": config.dry_run,
        "error": error,
        "prof_tools": sorted(VALID_PROF_TOOLS),
        "ssh_host": config.ssh_host or None,
        "remote_workdir": config.remote_workdir if config.mode == "ssh" else None,
        "local_script": str(config.local_script) if config.mode == "local" else None,
        "npu_device": config.npu_device,
        "chip": config.chip,
        "soc_version": config.soc_version,
        "example_command_msprof": build_command(payload) if enabled else None,
        "example_command_msprof_op": build_command(payload_op) if enabled else None,
        "script_options": script_options(),
        "default_script_path": TRIGGER_SCRIPTS[0]["id"],
    }


def attributes_to_cli_args(attributes: dict[str, Any], npu_device: int) -> list[str]:
    attrs = attributes or {}
    args = ["--device", str(npu_device)]
    int_fields = {
        "batch": "--batch",
        "query_heads": "--query-heads",
        "value_heads": "--value-heads",
        "tokens": "--tokens",
        "key_dim": "--key-dim",
        "value_dim": "--value-dim",
        "chunk_size": "--chunk-size",
        "mean_len": "--mean-len",
    }
    for key, flag in int_fields.items():
        if attrs.get(key) is not None:
            args.extend([flag, str(attrs[key])])
    if attrs.get("scale") is not None:
        args.extend(["--scale", str(attrs["scale"])])
    if attrs.get("dtype"):
        args.extend(["--dtype", str(attrs["dtype"])])
    if attrs.get("cu_seqlens"):
        args.extend(["--cu-seqlens", str(attrs["cu_seqlens"])])
    if attrs.get("varlen", True):
        args.append("--varlen")
    else:
        args.append("--no-varlen")
    return args


def build_prof_invocation(
    config: PerfRunnerConfig,
    *,
    prof_tool: str,
    output: str,
    script: str,
    py_args: list[str],
    kernel_name: str | None = None,
) -> str:
    py = " ".join(shlex.quote(part) for part in py_args)
    if prof_tool == "msprof":
        return f"msprof --output={shlex.quote(output)} python3 {shlex.quote(script)} {py}"
    parts = ["msprof", "op"]
    if prof_tool == "msprof_op_sim":
        parts.append("simulator")
        parts.append(f"--soc-version={shlex.quote(config.soc_version)}")
    parts.append(f"--output={shlex.quote(output)}")
    if kernel_name:
        parts.append(f"--kernel-name={shlex.quote(kernel_name)}")
    parts.append(f"python3 {shlex.quote(script)} {py}")
    return " ".join(parts)


def build_command(payload: dict[str, Any]) -> str:
    config = load_config()
    prof_tool = normalize_prof_tool(payload)
    attrs = payload.get("attributes") or {}
    kernel_name = str(payload.get("kernel_name") or "").strip() or None
    py_args = attributes_to_cli_args(attrs, resolve_npu_device(payload, config))
    output = str(prof_output_root(prof_tool, local=False))
    remote_script, local_script = resolve_script_paths(payload, config)
    invocation = build_prof_invocation(
        config,
        prof_tool=prof_tool,
        output=output,
        script=remote_script,
        py_args=py_args,
        kernel_name=kernel_name,
    )
    if config.mode == "ssh":
        remote = f"cd {shlex.quote(config.remote_workdir)} && {invocation}"
        return " ".join(shlex.quote(part) for part in _ssh_command(config, remote))
    local_output = str(prof_output_root(prof_tool, local=True))
    invocation = build_prof_invocation(
        config,
        prof_tool=prof_tool,
        output=local_output,
        script=str(local_script),
        py_args=py_args,
        kernel_name=kernel_name,
    )
    return invocation


def _ssh_command(config: PerfRunnerConfig, remote_command: str) -> list[str]:
    cmd = ["ssh"]
    if config.ssh_port:
        cmd.extend(["-p", config.ssh_port])
    if config.ssh_identity:
        cmd.extend(["-i", config.ssh_identity])
    cmd.extend(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"])
    cmd.append(f"{config.ssh_user}@{config.ssh_host}")
    cmd.append(remote_command)
    return cmd


def _scp_command(config: PerfRunnerConfig, remote_path: str, local_path: Path) -> list[str]:
    cmd = ["scp", "-r"]
    if config.ssh_port:
        cmd.extend(["-P", config.ssh_port])
    if config.ssh_identity:
        cmd.extend(["-i", config.ssh_identity])
    cmd.append(f"{config.ssh_user}@{config.ssh_host}:{remote_path}")
    cmd.append(str(local_path))
    return cmd


def _run_command(command: list[str] | str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        return subprocess.run(command, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _list_remote_prof_dirs(config: PerfRunnerConfig, prof_tool: str) -> set[str]:
    output = config.prof_output_op if prof_tool in {"msprof_op", "msprof_op_sim"} else config.prof_output_app
    prefix = prof_dir_prefix(prof_tool).lower()
    remote = f"ls -1 {shlex.quote(output)}/{prefix}* 2>/dev/null || true"
    result = _run_command(_ssh_command(config, remote))
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(Path(line).name)
    return names


def _list_local_prof_dirs(prof_tool: str) -> set[str]:
    root = prof_output_root(prof_tool, local=True)
    prefix = prof_dir_prefix(prof_tool)
    if not root.exists():
        return set()
    return {path.name for path in root.glob(f"{prefix}*") if path.is_dir()}


def _resolve_new_prof_dir(before: set[str], after: set[str], prof_tool: str) -> str:
    prefix = prof_dir_prefix(prof_tool)
    created = sorted(name for name in after - before if name.upper().startswith(prefix))
    if not created:
        raise RuntimeError(f"{prof_tool_label(prof_tool)} 执行完成，但未发现新的 {prefix}* 目录")
    return created[-1]


def _import_module(script_name: str):
    module_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载导入脚本：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    config = ensure_runner_configured()
    prof_tool = normalize_prof_tool(payload)
    command = build_command(payload)
    if config.dry_run:
        return {
            "status": "done",
            "message": "dry-run：未实际执行",
            "command": command,
            "prof_tool": prof_tool,
            "dry_run": True,
        }

    chip = resolve_chip(payload, config)
    model_id = payload.get("model_id") or "gdn"
    attrs = payload.get("attributes") or {}
    kernel_name = str(payload.get("kernel_name") or "").strip() or None
    operator_id = str(payload.get("operator_id") or "").strip() or None
    npu_device = resolve_npu_device(payload, config)
    remote_script, local_script = resolve_script_paths(payload, config)
    py_args = attributes_to_cli_args(attrs, npu_device)
    local_root = prof_output_root(prof_tool, local=True)
    remote_output = config.prof_output_op if prof_tool in {"msprof_op", "msprof_op_sim"} else config.prof_output_app

    if config.mode == "ssh":
        if not config.ssh_host:
            raise RuntimeError("PERF_SSH_HOST 未配置")
        before = _list_remote_prof_dirs(config, prof_tool)
        invocation = build_prof_invocation(
            config,
            prof_tool=prof_tool,
            output=remote_output,
            script=remote_script,
            py_args=py_args,
            kernel_name=kernel_name,
        )
        remote = f"cd {shlex.quote(config.remote_workdir)} && {invocation}"
        _run_command(_ssh_command(config, remote))
        after = _list_remote_prof_dirs(config, prof_tool)
        prof_name = _resolve_new_prof_dir(before, after, prof_tool)
        local_dir = local_root / prof_name
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        remote_prof = f"{remote_output.rstrip('/')}/{prof_name}"
        if local_dir.exists():
            import shutil

            shutil.rmtree(local_dir)
        _run_command(_scp_command(config, remote_prof, local_dir.parent))
        prof_dir = local_dir
    elif config.mode == "local":
        before = _list_local_prof_dirs(prof_tool)
        script = local_script
        local_root.mkdir(parents=True, exist_ok=True)
        invocation_parts = shlex.split(
            build_prof_invocation(
                config,
                prof_tool=prof_tool,
                output=str(local_root),
                script=str(script),
                py_args=py_args,
                kernel_name=kernel_name,
            ),
            posix=(os.name != "nt"),
        )
        try:
            _run_command(invocation_parts, cwd=ROOT)
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 msprof，请在 NPU 主机上执行，或配置 PERF_RUN_MODE=ssh") from exc
        after = _list_local_prof_dirs(prof_tool)
        prof_name = _resolve_new_prof_dir(before, after, prof_tool)
        prof_dir = local_root / prof_name
    else:
        raise RuntimeError(f"不支持的执行模式：{config.mode}")

    if prof_tool == "msprof":
        import_module = _import_module("import_prof_gdr.py")
        data = import_module.import_prof(prof_dir, model_id, chip, replace_mock=False, device_id=npu_device)
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)
        snapshot["prof_tool"] = prof_tool
    else:
        import_module = _import_module("import_msprof_op.py")
        data = import_module.import_msprof_op(
            prof_dir,
            model_id,
            chip,
            attributes=attrs,
            kernel_name=kernel_name,
            operator_id=operator_id,
            prof_tool=prof_tool,
            device_id=npu_device,
        )
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)

    data["runs"] = [
        item
        for item in data.get("runs", [])
        if not (item.get("created_by") in {"import_prof_gdr", "import_msprof_op"} and item.get("snapshot_id") == snapshot["id"])
    ]
    for path in import_module.PERF_PATHS:
        path.write_text(
            __import__("json").dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "done",
        "message": f"{prof_tool_label(prof_tool)} 执行并导入：{prof_dir.name}",
        "command": command,
        "prof_tool": prof_tool,
        "prof_dir": str(prof_dir),
        "snapshot": snapshot,
        "data": data,
    }
