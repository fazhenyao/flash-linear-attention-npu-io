from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROF_APP_ROOT = ROOT / "data" / "prof_gdr"
PROF_OP_ROOT = ROOT / "data" / "prof_op"
PROF_SOURCE_PATTERN = re.compile(r"^(OPPROF_|PROF_)", re.IGNORECASE)
MAX_PROF_UPLOAD_BYTES = 512 * 1024 * 1024
VALID_PROF_TOOLS = {"msprof", "msprof_op", "msprof_op_sim"}
ATTR_DEFAULTS = {
    "batch": 1,
    "query_heads": 32,
    "value_heads": 32,
    "tokens": 4087,
    "key_dim": 128,
    "value_dim": 128,
    "chunk_size": 64,
    "mean_len": 1024,
    "dtype": "bf16",
    "varlen": True,
}

DEFAULT_TRIGGER_SCRIPT = "scripts/flash_gated_delta_rule.py"
LOCAL_PROF_OUTPUT_APP = "data/prof_gdr"
LOCAL_PROF_OUTPUT_OP = "data/prof_op"

TRIGGER_SCRIPTS = [
    {
        "id": DEFAULT_TRIGGER_SCRIPT,
        "label": DEFAULT_TRIGGER_SCRIPT,
        "remote": DEFAULT_TRIGGER_SCRIPT,
        "local": DEFAULT_TRIGGER_SCRIPT,
    },
]

LEGACY_TRIGGER_SCRIPT_IDS = {
    "flash-linear-attention-npu/examples/flash_gated_delta_rule.py": DEFAULT_TRIGGER_SCRIPT,
    "ref/flash_gated_delta_rule.py": DEFAULT_TRIGGER_SCRIPT,
    "examples/flash_gated_delta_rule.py": DEFAULT_TRIGGER_SCRIPT,
}


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
        remote_workdir=os.environ.get("PERF_REMOTE_WORKDIR", "").strip() or ".",
        remote_script=os.environ.get("PERF_REMOTE_SCRIPT", "").strip() or DEFAULT_TRIGGER_SCRIPT,
        local_script=Path(os.environ.get("PERF_LOCAL_SCRIPT", DEFAULT_TRIGGER_SCRIPT)),
        npu_device=int(os.environ.get("PERF_NPU_DEVICE", "2")),
        chip=os.environ.get("PERF_CHIP", "").strip().upper() or "A2",
        prof_output_app=os.environ.get("PERF_PROF_OUTPUT", LOCAL_PROF_OUTPUT_APP).strip() or LOCAL_PROF_OUTPUT_APP,
        prof_output_op=os.environ.get("PERF_OP_OUTPUT", LOCAL_PROF_OUTPUT_OP).strip() or LOCAL_PROF_OUTPUT_OP,
        soc_version=os.environ.get("PERF_SOC_VERSION", "").strip() or "Ascend910B",
        dry_run=_env_bool("PERF_RUN_DRY_RUN"),
    )


def to_repo_relative_path(path: Path | str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def local_prof_output_path(prof_tool: str) -> str:
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        return to_repo_relative_path(LOCAL_PROF_OUTPUT_OP)
    return to_repo_relative_path(LOCAL_PROF_OUTPUT_APP)


def resolve_script_paths(payload: dict[str, Any], config: PerfRunnerConfig) -> tuple[str, str]:
    script_path = str(payload.get("script_path") or "").strip() or TRIGGER_SCRIPTS[0]["id"]
    script_path = LEGACY_TRIGGER_SCRIPT_IDS.get(script_path, script_path)
    entry = next(
        (item for item in TRIGGER_SCRIPTS if item["id"] == script_path or item["label"] == script_path),
        None,
    )
    if entry is None:
        allowed = ", ".join(item["label"] for item in TRIGGER_SCRIPTS)
        raise ValueError(f"未知脚本路径：{script_path}（可选：{allowed}）")
    configured = config.local_script or Path(entry["local"])
    local_abs = configured if configured.is_absolute() else ROOT / configured
    if not local_abs.exists():
        raise FileNotFoundError(f"本地脚本不存在：{local_abs}")
    return entry["remote"], to_repo_relative_path(local_abs)


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
        "msprof_op": "msopprof（算子）",
        "msprof_op_sim": "msopprof simulator（仿真）",
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
    payload_op = {
        **payload,
        "prof_tool": "msprof_op",
        "kernel_name": "chunk_bwd_dqkwg",
        "warm_up": resolve_op_warm_up({"prof_tool": "msprof_op"}),
        "launch_count": resolve_op_launch_count({"prof_tool": "msprof_op"}),
    }
    return {
        "enabled": enabled,
        "mode": config.mode,
        "dry_run": config.dry_run,
        "error": error,
        "prof_tools": sorted(VALID_PROF_TOOLS),
        "ssh_host": config.ssh_host or None,
        "remote_workdir": config.remote_workdir if config.mode == "ssh" else None,
        "local_script": to_repo_relative_path(config.local_script) if config.mode == "local" else None,
        "npu_device": config.npu_device,
        "chip": config.chip,
        "soc_version": config.soc_version,
        "op_warm_up": resolve_op_warm_up({"prof_tool": "msprof_op"}),
        "op_launch_count": resolve_op_launch_count({"prof_tool": "msprof_op"}),
        "example_command_msprof": build_command(payload) if enabled else None,
        "example_command_msprof_op": build_command(payload_op) if enabled else None,
        "script_options": script_options(),
        "default_script_path": TRIGGER_SCRIPTS[0]["id"],
    }


def normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    attrs = dict(attributes or {})
    for key, default in ATTR_DEFAULTS.items():
        value = attrs.get(key)
        if key == "dtype":
            if not value:
                attrs[key] = default
            continue
        if key == "varlen":
            if value is None:
                attrs[key] = default
            continue
        if value is None or value == "":
            attrs[key] = default
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            attrs[key] = default
            continue
        if numeric <= 0:
            attrs[key] = default
    if attrs.get("scale") in (None, "", 0):
        key_dim = int(attrs.get("key_dim") or ATTR_DEFAULTS["key_dim"])
        attrs["scale"] = key_dim ** -0.5
    return attrs


def attributes_to_cli_args(attributes: dict[str, Any], npu_device: int) -> list[str]:
    attrs = normalize_attributes(attributes)
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


def split_kernel_names(kernel_name: str | None) -> list[str]:
    if not kernel_name:
        return []
    return [part.strip() for part in re.split(r"[|,;\n]+", kernel_name) if part.strip()]


def format_kernel_name_arg(kernel_name: str | None) -> str | None:
    names = split_kernel_names(kernel_name)
    if not names:
        return None
    return "|".join(names)


def single_kernel_name_override(kernel_name: str | None) -> str | None:
    names = split_kernel_names(kernel_name)
    return names[0] if len(names) == 1 else None


def resolve_op_warm_up(payload: dict[str, Any]) -> int | None:
    prof_tool = str(payload.get("prof_tool") or "msprof").strip()
    if prof_tool not in {"msprof_op", "msprof_op_sim"}:
        return None
    raw = payload.get("warm_up")
    if raw is not None and str(raw).strip() != "":
        return max(0, int(raw))
    env = os.environ.get("PERF_OP_WARM_UP", "10").strip()
    return int(env) if env else None


def resolve_op_launch_count(payload: dict[str, Any]) -> int | None:
    prof_tool = str(payload.get("prof_tool") or "msprof").strip()
    if prof_tool not in {"msprof_op", "msprof_op_sim"}:
        return None
    raw = payload.get("launch_count")
    if raw is not None and str(raw).strip() != "":
        return max(1, int(raw))
    env = os.environ.get("PERF_OP_LAUNCH_COUNT", "10").strip()
    return int(env) if env else None


def build_prof_invocation(
    config: PerfRunnerConfig,
    *,
    prof_tool: str,
    output: str,
    script: str,
    py_args: list[str],
    kernel_name: str | None = None,
    warm_up: int | None = None,
    launch_count: int | None = None,
) -> str:
    py = " ".join(shlex.quote(part) for part in py_args)
    if prof_tool == "msprof":
        return f"msprof --output={shlex.quote(output)} python3 {shlex.quote(script)} {py}"
    parts = ["msopprof"]
    if prof_tool == "msprof_op_sim":
        parts.append("simulator")
        parts.append(f"--soc-version={shlex.quote(config.soc_version)}")
    if warm_up is not None:
        parts.append(f"--warm-up={warm_up}")
    if launch_count is not None:
        parts.append(f"--launch-count={launch_count}")
    parts.append(f"--output={shlex.quote(output)}")
    kernel_arg = format_kernel_name_arg(kernel_name)
    if kernel_arg:
        parts.append(f"--kernel-name={shlex.quote(kernel_arg)}")
    parts.append(f"python3 {shlex.quote(script)} {py}")
    return " ".join(parts)


def build_command(payload: dict[str, Any]) -> str:
    config = load_config()
    prof_tool = normalize_prof_tool(payload)
    attrs = payload.get("attributes") or {}
    kernel_name = str(payload.get("kernel_name") or "").strip() or None
    warm_up = resolve_op_warm_up(payload)
    launch_count = resolve_op_launch_count(payload)
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
        warm_up=warm_up,
        launch_count=launch_count,
    )
    if config.mode == "ssh":
        remote = f"cd {shlex.quote(config.remote_workdir)} && {invocation}"
        return " ".join(shlex.quote(part) for part in _ssh_command(config, remote))
    local_output = local_prof_output_path(prof_tool)
    invocation = build_prof_invocation(
        config,
        prof_tool=prof_tool,
        output=local_output,
        script=local_script,
        py_args=py_args,
        kernel_name=kernel_name,
        warm_up=warm_up,
        launch_count=launch_count,
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
    warm_up = resolve_op_warm_up(payload)
    launch_count = resolve_op_launch_count(payload)
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
            warm_up=warm_up,
            launch_count=launch_count,
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
                output=local_prof_output_path(prof_tool),
                script=script,
                py_args=py_args,
                kernel_name=kernel_name,
                warm_up=warm_up,
                launch_count=launch_count,
            ),
            posix=(os.name != "nt"),
        )
        try:
            _run_command(invocation_parts, cwd=ROOT)
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 msopprof，请在 NPU 主机上执行，或配置 PERF_RUN_MODE=ssh") from exc
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
            operator_id=single_kernel_name_override(operator_id or kernel_name),
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
        "prof_source": prof_dir.name,
        "snapshot": snapshot,
        "data": data,
    }


def resolve_prof_dir_path(raw: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("prof_dir required")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("prof_dir must be under project root") from exc
    if not path.is_dir():
        raise ValueError(f"prof_dir not found: {path}")
    return path


def infer_prof_tool_from_dir(prof_dir: Path, prof_tool: str | None = None) -> str:
    if prof_tool:
        normalized = str(prof_tool).strip()
        if normalized not in VALID_PROF_TOOLS:
            raise ValueError(f"prof_tool must be one of {sorted(VALID_PROF_TOOLS)}")
        return normalized
    name = prof_dir.name.upper()
    if name.startswith("OPPROF_"):
        return "msprof_op"
    if name.startswith("PROF_"):
        return "msprof"
    parent = prof_dir.parent.resolve()
    if parent == PROF_OP_ROOT.resolve():
        return "msprof_op"
    if parent == PROF_APP_ROOT.resolve():
        return "msprof"
    raise ValueError("无法识别 prof 类型，请使用 OPPROF_* 或 PROF_* 目录")


def import_prof_directory(payload: dict[str, Any]) -> dict[str, Any]:
    prof_dir = resolve_prof_dir_path(str(payload.get("prof_dir") or ""))
    prof_tool = infer_prof_tool_from_dir(prof_dir, payload.get("prof_tool"))
    config = load_config()
    chip = resolve_chip(payload, config)
    model_id = str(payload.get("model_id") or "gdn").strip() or "gdn"
    attrs = payload.get("attributes") or {}
    kernel_name = str(payload.get("kernel_name") or "").strip() or None
    operator_id = str(payload.get("operator_id") or "").strip() or None
    npu_device = resolve_npu_device(payload, config)

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
            operator_id=single_kernel_name_override(operator_id or kernel_name),
            prof_tool=prof_tool,
            device_id=npu_device,
        )
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)

    data["runs"] = [
        item
        for item in data.get("runs", [])
        if not (
            item.get("created_by") in {"import_prof_gdr", "import_msprof_op"}
            and item.get("snapshot_id") == snapshot["id"]
        )
    ]
    for path in import_module.PERF_PATHS:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "data": data,
        "snapshot": snapshot,
        "prof_dir": str(prof_dir),
        "prof_source": prof_dir.name,
        "prof_tool": prof_tool,
        "case_id": snapshot.get("case_id"),
        "message": f"{prof_tool_label(prof_tool)} 目录导入：{prof_dir.name}",
    }


def list_prof_directories() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for prof_dir in sorted(PROF_OP_ROOT.glob("OPPROF_*")):
        if prof_dir.is_dir():
            entries.append({
                "prof_dir": f"data/prof_op/{prof_dir.name}",
                "prof_source": prof_dir.name,
                "prof_tool": "msprof_op",
                "kind": "msopprof",
            })
    for prof_dir in sorted(PROF_APP_ROOT.glob("PROF_*")):
        if prof_dir.is_dir():
            entries.append({
                "prof_dir": f"data/prof_gdr/{prof_dir.name}",
                "prof_source": prof_dir.name,
                "prof_tool": "msprof",
                "kind": "msprof",
            })
    entries.sort(key=lambda item: item["prof_source"], reverse=True)
    return entries


def match_prof_dir_from_paths(sample_paths: list[str]) -> str:
    normalized = [str(item or "").strip().replace("\\", "/") for item in sample_paths if str(item or "").strip()]
    if not normalized:
        raise ValueError("sample_paths required")
    top = normalized[0].split("/")[0]
    if top.upper().startswith("OPPROF_"):
        return f"data/prof_op/{top}"
    if top.upper().startswith("PROF_"):
        return f"data/prof_gdr/{top}"
    for entry in list_prof_directories():
        root = resolve_prof_dir_path(entry["prof_dir"])
        if any((root / rel_path).exists() for rel_path in normalized):
            return entry["prof_dir"]
    raise ValueError("无法匹配 Prof 目录，请选择 OPPROF_* 或 PROF_* 目录")


def sanitize_upload_rel_path(rel_path: str) -> str:
    rel = str(rel_path or "").strip().replace("\\", "/").lstrip("/")
    parts = [part for part in rel.split("/") if part and part not in {".", ".."}]
    if not parts or any(part == ".." for part in rel.split("/")):
        raise ValueError(f"invalid upload path: {rel_path}")
    return "/".join(parts)


def extract_prof_source_from_paths(paths: list[str], explicit: str = "") -> str:
    explicit_name = str(explicit or "").strip()
    if explicit_name:
        if not PROF_SOURCE_PATTERN.match(explicit_name):
            raise ValueError("prof_source must start with OPPROF_ or PROF_")
        return explicit_name
    for path in paths:
        for part in sanitize_upload_rel_path(path).split("/"):
            if PROF_SOURCE_PATTERN.match(part):
                return part
    raise ValueError("无法识别 OPPROF_* 或 PROF_* 目录名，请上传正确目录或填写目录名")


def destination_for_prof_source(prof_source: str) -> Path:
    if prof_source.upper().startswith("OPPROF_"):
        return PROF_OP_ROOT / prof_source
    if prof_source.upper().startswith("PROF_"):
        return PROF_APP_ROOT / prof_source
    raise ValueError("prof_source must start with OPPROF_ or PROF_")


def normalize_upload_entry_path(rel_path: str, prof_source: str) -> str:
    rel = sanitize_upload_rel_path(rel_path)
    parts = rel.split("/")
    if parts and parts[0] == prof_source:
        parts = parts[1:]
    return "/".join(parts)


def save_uploaded_prof_files(entries: list[tuple[str, bytes]], *, prof_source: str = "") -> Path:
    if not entries:
        raise ValueError("upload files required")
    total_size = sum(len(content) for _, content in entries)
    if total_size > MAX_PROF_UPLOAD_BYTES:
        raise ValueError(f"upload too large (> {MAX_PROF_UPLOAD_BYTES // (1024 * 1024)} MB)")
    source = extract_prof_source_from_paths([path for path, _ in entries], prof_source)
    dest_root = destination_for_prof_source(source)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    wrote = False
    for rel_path, content in entries:
        normalized = normalize_upload_entry_path(rel_path, source)
        if not normalized:
            continue
        target = dest_root / normalized
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        wrote = True
    if not wrote:
        raise ValueError("upload did not contain any files")
    return dest_root


def save_uploaded_prof_zip(zip_bytes: bytes, *, prof_source: str = "") -> Path:
    if len(zip_bytes) > MAX_PROF_UPLOAD_BYTES:
        raise ValueError(f"upload too large (> {MAX_PROF_UPLOAD_BYTES // (1024 * 1024)} MB)")
    entries: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.filename.startswith("__MACOSX/"):
                continue
            entries.append((info.filename, archive.read(info)))
    if not entries:
        raise ValueError("zip archive is empty")
    return save_uploaded_prof_files(entries, prof_source=prof_source)


def ingest_uploaded_prof(
    *,
    archive: bytes | None = None,
    files: list[tuple[str, bytes]] | None = None,
    prof_source: str = "",
) -> dict[str, str]:
    if archive:
        dest_root = save_uploaded_prof_zip(archive, prof_source=prof_source)
    elif files:
        dest_root = save_uploaded_prof_files(files, prof_source=prof_source)
    else:
        raise ValueError("archive or files required")
    try:
        dest_root.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("upload destination must be under project root") from exc
    if dest_root.parent.resolve() not in {PROF_OP_ROOT.resolve(), PROF_APP_ROOT.resolve()}:
        raise ValueError("upload destination must be under data/prof_op or data/prof_gdr")
    return {
        "prof_dir": str(dest_root.relative_to(ROOT)),
        "prof_source": dest_root.name,
    }


def run_snapshot(run: dict[str, Any], data: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = run.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot
    snapshot_id = str(run.get("snapshot_id") or "").strip()
    if not snapshot_id:
        return None
    return next((item for item in data.get("snapshots", []) if item.get("id") == snapshot_id), None)


def infer_prof_tool(snapshot: dict[str, Any], prof_source: str) -> str:
    tool = str(snapshot.get("prof_tool") or "").strip()
    if tool:
        return tool
    if str(prof_source or "").upper().startswith("OPPROF_"):
        return "msprof_op"
    return "msprof"


def primary_snapshot_for_case(data: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    snapshots = [
        item
        for item in data.get("snapshots", [])
        if item.get("case_id") == case_id
    ]
    if not snapshots:
        return None
    return sorted(snapshots, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0]


def collect_case_csv_files(prof_dir: Path, prof_tool: str) -> list[Path]:
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        return sorted(path for path in prof_dir.rglob("*.csv") if path.is_file())
    output_dir = prof_dir / "mindstudio_profiler_output"
    if output_dir.is_dir():
        return sorted(output_dir.glob("*.csv"))
    return sorted(path for path in prof_dir.rglob("*.csv") if path.is_file())


def case_export_slug(case: dict[str, Any], snapshot: dict[str, Any]) -> str:
    prof_source = str(snapshot.get("prof_source") or "").strip()
    prof_tool = infer_prof_tool(snapshot, prof_source)
    tool = "msopprof" if prof_tool in {"msprof_op", "msprof_op_sim"} else "msprof"

    case_id = str(case.get("id") or "")
    time_match = re.search(r"(\d{8})t(\d{6})", case_id)
    time_part = f"{time_match.group(1)}-{time_match.group(2)}" if time_match else case_id[-12:]

    parts = [tool, time_part]
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        kernel = str(snapshot.get("kernel_name") or "").strip()
        if kernel:
            full_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", kernel).strip("-")
            if full_name:
                parts.append(full_name)

    if prof_source:
        parts.append(prof_source.split("_")[-1][:10])

    slug = "-".join(part for part in parts if part)
    return re.sub(r"-+", "-", slug).strip("-")[:240]


def build_perf_cases_csv_download(data: dict[str, Any], case_ids: list[str]) -> tuple[str, bytes]:
    import io
    import json
    import zipfile

    wanted = [str(case_id).strip() for case_id in case_ids if str(case_id).strip()]
    if not wanted:
        raise ValueError("case ids required")

    entries: list[tuple[dict[str, Any], dict[str, Any], Path, list[Path]]] = []
    for case_id in wanted:
        case = next((item for item in data.get("cases", []) if item.get("id") == case_id), None)
        if case is None:
            raise ValueError(f"case not found: {case_id}")
        snapshot = primary_snapshot_for_case(data, case_id)
        if snapshot is None:
            raise ValueError(f"case has no snapshot: {case_id}")
        prof_source = str(snapshot.get("prof_source") or "").strip()
        if not prof_source:
            raise ValueError(f"snapshot missing prof_source: {case_id}")
        prof_tool = infer_prof_tool(snapshot, prof_source)
        prof_dir = find_prof_dir(prof_output_root(prof_tool, local=True), prof_source)
        if prof_dir is None:
            raise ValueError(f"prof dir not found: {prof_source}")
        csv_files = collect_case_csv_files(prof_dir, prof_tool)
        if not csv_files:
            raise ValueError(f"no csv files under {prof_dir}")
        entries.append((case, snapshot, prof_dir, csv_files))

    slugs = [case_export_slug(case, snapshot) for case, snapshot, _, _ in entries]
    if len(entries) == 1:
        filename = f"perf-csv-{slugs[0]}.zip"
    else:
        filename = f"perf-csv-bundle-{len(entries)}-{'_'.join(slugs[:2])}"
        if len(slugs) > 2:
            filename += f"_plus{len(slugs) - 2}"
        filename = f"{filename[:240]}.zip"
    buffer = io.BytesIO()
    manifest_cases: list[dict[str, Any]] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for (case, snapshot, prof_dir, csv_files), export_slug in zip(entries, slugs):
            archive_root = f"{export_slug}/{prof_dir.name}"
            manifest_cases.append({
                "case_id": case["id"],
                "export_slug": export_slug,
                "case_label": case.get("label"),
                "snapshot_id": snapshot.get("id"),
                "prof_source": prof_dir.name,
                "prof_tool": infer_prof_tool(snapshot, prof_dir.name),
                "kernel_name": snapshot.get("kernel_name") or "",
                "csv_files": [path.relative_to(prof_dir).as_posix() for path in csv_files],
            })
            for path in csv_files:
                archive.write(path, f"{archive_root}/{path.relative_to(prof_dir).as_posix()}")
        archive.writestr(
            "manifest.json",
            json.dumps({"kind": "perf-case-csv-bundle", "cases": manifest_cases}, ensure_ascii=False, indent=2) + "\n",
        )
    return filename, buffer.getvalue()


def find_prof_dir(root: Path, prof_source: str) -> Path | None:
    source = str(prof_source or "").strip()
    if not source:
        return None
    direct = root / source
    if direct.is_dir():
        return direct
    lowered = source.lower()
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == lowered:
            return child
    return None


def resolve_run_prof_dir(run: dict[str, Any], data: dict[str, Any]) -> Path | None:
    stored = str(run.get("prof_dir") or "").strip()
    if stored:
        path = Path(stored)
        if path.is_dir():
            return path

    snapshot = run_snapshot(run, data)
    prof_source = str(run.get("prof_source") or (snapshot or {}).get("prof_source") or "").strip()
    if not prof_source:
        return None

    prof_tool = infer_prof_tool(snapshot or {}, prof_source) if snapshot else str(run.get("prof_tool") or "msprof").strip()
    return find_prof_dir(prof_output_root(prof_tool, local=True), prof_source)


def build_perf_run_download(run: dict[str, Any], data: dict[str, Any]) -> tuple[str, bytes]:
    import io
    import json
    import zipfile

    if run.get("status") != "done":
        raise ValueError("仅已完成的执行记录可下载")

    prof_dir = resolve_run_prof_dir(run, data)
    if prof_dir is None:
        raise ValueError("未找到对应的 profiling 输出目录")

    snapshot = run_snapshot(run, data)
    summary = {
        "run": {key: value for key, value in run.items() if key != "snapshot"},
        "snapshot": snapshot,
        "prof_dir": str(prof_dir),
    }
    filename = f"{run.get('id', 'run')}-{prof_dir.name}.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run-summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        for path in sorted(prof_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, f"{prof_dir.name}/{path.relative_to(prof_dir).as_posix()}")
    return filename, buffer.getvalue()
