from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROF_APP_ROOT = ROOT / "data" / "prof_gdr"
PROF_OP_ROOT = ROOT / "data" / "prof_op"
PROF_SOURCE_PATTERN = re.compile(r"^(OPPROF_|PROF_)", re.IGNORECASE)
MAX_PROF_UPLOAD_BYTES = 512 * 1024 * 1024
VALID_PROF_TOOLS = {"msprof", "msprof_op", "msprof_op_sim"}
VALID_TASK_TYPES = {"profile", "build_install"}
VALID_CHIPS = {"A2", "A3", "A5"}
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
    remote_env_script: str
    remote_path_prepend: str
    remote_conda_sh: str
    remote_conda_env: str
    remote_source_repo: str
    allowed_cann_roots: tuple[str, ...]
    allowed_source_roots: tuple[str, ...]
    remote_build_root: str
    local_script: Path
    npu_device: int
    chip: str
    prof_output_app: str
    prof_output_op: str
    local_prof_output_app: str
    local_prof_output_op: str
    soc_version: str
    dry_run: bool


@dataclass(frozen=True)
class ExecutionEnvironment:
    customized: bool
    cann_path: str
    env_script: str
    conda_env: str
    source_repo: str
    rebuild: bool
    branch: str
    branch_source: str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_paths(name: str) -> tuple[str, ...]:
    return tuple(item.strip().rstrip("/") for item in os.environ.get(name, "").split(";") if item.strip())


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
        remote_env_script=os.environ.get("PERF_REMOTE_ENV_SCRIPT", "").strip(),
        remote_path_prepend=os.environ.get("PERF_REMOTE_PATH_PREPEND", "").strip(),
        remote_conda_sh=os.environ.get("PERF_REMOTE_CONDA_SH", "").strip(),
        remote_conda_env=os.environ.get("PERF_REMOTE_CONDA_ENV", "").strip(),
        remote_source_repo=os.environ.get("PERF_REMOTE_SOURCE_REPO", "").strip().rstrip("/"),
        allowed_cann_roots=_env_paths("PERF_ALLOWED_CANN_ROOTS"),
        allowed_source_roots=_env_paths("PERF_ALLOWED_SOURCE_ROOTS"),
        remote_build_root=os.environ.get("PERF_REMOTE_BUILD_ROOT", "/tmp/fla-runner-builds").strip().rstrip("/"),
        local_script=Path(os.environ.get("PERF_LOCAL_SCRIPT", DEFAULT_TRIGGER_SCRIPT)),
        npu_device=int(os.environ.get("PERF_NPU_DEVICE", "2")),
        chip=os.environ.get("PERF_CHIP", "").strip().upper() or "A2",
        prof_output_app=os.environ.get("PERF_PROF_OUTPUT", LOCAL_PROF_OUTPUT_APP).strip() or LOCAL_PROF_OUTPUT_APP,
        prof_output_op=os.environ.get("PERF_OP_OUTPUT", LOCAL_PROF_OUTPUT_OP).strip() or LOCAL_PROF_OUTPUT_OP,
        local_prof_output_app=os.environ.get("PERF_LOCAL_PROF_OUTPUT", LOCAL_PROF_OUTPUT_APP).strip() or LOCAL_PROF_OUTPUT_APP,
        local_prof_output_op=os.environ.get("PERF_LOCAL_OP_OUTPUT", LOCAL_PROF_OUTPUT_OP).strip() or LOCAL_PROF_OUTPUT_OP,
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


def _normalized_remote_absolute_path(value: str, field: str) -> str:
    text = str(value or "").strip().rstrip("/")
    path = PurePosixPath(text)
    if not text.startswith("/") or len(text) > 500 or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{field} 必须为不含路径穿越的绝对路径")
    if not re.fullmatch(r"/[A-Za-z0-9._+@/-]+", text):
        raise ValueError(f"{field} 包含不支持的字符")
    return path.as_posix().rstrip("/")


def _path_allowed(path: str, roots: tuple[str, ...]) -> bool:
    candidate = PurePosixPath(path)
    for raw_root in roots:
        root = PurePosixPath(raw_root)
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _valid_source_branch(branch: str) -> bool:
    return bool(
        branch
        and len(branch) <= 200
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
        and ".." not in branch
        and "//" not in branch
        and "@{" not in branch
        and not branch.endswith(("/", ".", ".lock"))
    )


def _parse_source_branch_refs(output: str) -> list[dict[str, str]]:
    branches: set[tuple[str, str]] = set()
    for raw in output.splitlines():
        ref = raw.strip()
        if ref.startswith("refs/heads/"):
            source, name = "local", ref.removeprefix("refs/heads/")
        elif ref.startswith("refs/remotes/origin/"):
            source, name = "remote", ref.removeprefix("refs/remotes/origin/")
        else:
            continue
        if name != "HEAD" and _valid_source_branch(name):
            branches.add((source, name))
    ordered = sorted(branches, key=lambda item: (item[0] != "local", item[1].casefold()))
    return [{"source": source, "name": name} for source, name in ordered[:400]]


def list_remote_source_branches(config: PerfRunnerConfig, source_repo: str) -> list[dict[str, str]]:
    if config.mode != "ssh":
        raise ValueError("源码分支查询仅支持 SSH Relay")
    repo = _normalized_remote_absolute_path(source_repo, "源码仓库路径")
    roots = config.allowed_source_roots or (
        (str(PurePosixPath(config.remote_source_repo).parent),) if config.remote_source_repo else ()
    )
    if not roots or not _path_allowed(repo, roots):
        raise ValueError("源码仓库路径不在 Relay 允许目录内")
    list_command = (
        f"test -d {shlex.quote(repo)}/.git"
        f" && git -C {shlex.quote(repo)} for-each-ref "
        f"--format={shlex.quote('%(refname)')} refs/heads refs/remotes/origin"
    )
    result = _run_remote_checked(config, list_command, "源码分支查询")
    branches = _parse_source_branch_refs(result.stdout)
    refresh_command = (
        "export GIT_TERMINAL_PROMPT=0; "
        f"timeout --signal=TERM --kill-after=2s 8s git -C {shlex.quote(repo)} "
        "fetch --prune origin >/dev/null 2>&1"
    )
    try:
        _run_remote_checked(config, refresh_command, "远程源码分支刷新")
        refreshed = _run_remote_checked(config, list_command, "源码分支查询")
        branches = _parse_source_branch_refs(refreshed.stdout) or branches
    except RuntimeError:
        pass
    if not branches:
        raise RuntimeError("源码仓库没有可用的本地或远程分支")
    return branches


def configured_cann_path(config: PerfRunnerConfig) -> str:
    script = config.remote_env_script.rstrip("/")
    return str(PurePosixPath(script).parent) if script.endswith("/set_env.sh") else script


def execution_environment_defaults(config: PerfRunnerConfig) -> dict[str, Any]:
    return {
        "cann_path": configured_cann_path(config),
        "conda_env": config.remote_conda_env,
        "source_repo": config.remote_source_repo,
        "rebuild": False,
        "branch": "",
        "branch_source": "local",
    }


def execution_environment_summary(execution: ExecutionEnvironment) -> dict[str, Any]:
    return {
        "cann_path": execution.cann_path,
        "conda_env": execution.conda_env,
        "source_repo": execution.source_repo,
        "rebuild": execution.rebuild,
        "branch": execution.branch,
        "branch_source": execution.branch_source,
    }


def resolve_execution_environment(payload: dict[str, Any], config: PerfRunnerConfig) -> ExecutionEnvironment:
    raw = payload.get("execution_environment")
    customized = raw is not None
    if customized and not isinstance(raw, dict):
        raise ValueError("execution_environment 必须为对象")
    value = dict(raw or execution_environment_defaults(config))
    raw_cann_path = str(value.get("cann_path") or configured_cann_path(config)).strip()
    cann_path = _normalized_remote_absolute_path(raw_cann_path, "CANN 路径") if raw_cann_path else ""
    if cann_path and cann_path.endswith("/set_env.sh"):
        env_script = cann_path
        cann_path = str(PurePosixPath(cann_path).parent)
    else:
        env_script = f"{cann_path}/set_env.sh" if cann_path else ""
    raw_source_repo = str(value.get("source_repo") or config.remote_source_repo).strip()
    source_repo = (
        _normalized_remote_absolute_path(raw_source_repo, "源码仓库路径")
        if raw_source_repo
        else ""
    )
    conda_env = str(value.get("conda_env") or config.remote_conda_env).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", conda_env):
        raise ValueError("Conda 环境名称不合法")
    branch = str(value.get("branch") or "").strip()
    branch_source = str(value.get("branch_source") or "local").strip().lower()
    if branch_source not in {"local", "remote"}:
        raise ValueError("源码分支来源必须为 local 或 remote")
    raw_rebuild = value.get("rebuild", False)
    if not isinstance(raw_rebuild, bool):
        raise ValueError("rebuild 必须为布尔值")
    rebuild = raw_rebuild
    if rebuild and not branch:
        raise ValueError("重新编译安装时必须指定分支")
    if branch and not _valid_source_branch(branch):
        raise ValueError("源码分支名称不合法")

    cann_roots = config.allowed_cann_roots or ((configured_cann_path(config),) if configured_cann_path(config) else ())
    source_roots = config.allowed_source_roots or (
        (str(PurePosixPath(config.remote_source_repo).parent),) if config.remote_source_repo else ()
    )
    if customized and (not cann_path or not cann_roots or not _path_allowed(cann_path, cann_roots)):
        raise ValueError("CANN 路径不在 Relay 允许目录内")
    if customized and (not source_repo or not source_roots or not _path_allowed(source_repo, source_roots)):
        raise ValueError("源码仓库路径不在 Relay 允许目录内")
    return ExecutionEnvironment(
        customized=customized,
        cann_path=cann_path,
        env_script=env_script,
        conda_env=conda_env,
        source_repo=source_repo,
        rebuild=rebuild,
        branch=branch,
        branch_source=branch_source,
    )


def local_prof_output_path(prof_tool: str, config: PerfRunnerConfig | None = None) -> str:
    config = config or load_config()
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        return to_repo_relative_path(config.local_prof_output_op)
    return to_repo_relative_path(config.local_prof_output_app)


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
    # The task selects a whitelisted script ID; the trusted Relay config may map
    # that ID to a different path on its NPU host.
    remote_script = config.remote_script or entry["remote"]
    return remote_script, to_repo_relative_path(local_abs)


def script_options() -> list[dict[str, str]]:
    return [{"id": item["id"], "label": item["label"]} for item in TRIGGER_SCRIPTS]


def normalize_prof_tool(payload: dict[str, Any]) -> str:
    prof_tool = str(payload.get("prof_tool") or "msprof").strip()
    if prof_tool not in VALID_PROF_TOOLS:
        raise ValueError(f"prof_tool must be one of {sorted(VALID_PROF_TOOLS)}")
    return prof_tool


def normalize_task_type(payload: dict[str, Any]) -> str:
    task_type = str(payload.get("task_type") or "profile").strip()
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {sorted(VALID_TASK_TYPES)}")
    return task_type


def prof_output_root(prof_tool: str, *, local: bool) -> Path:
    config = load_config()
    if prof_tool in {"msprof_op", "msprof_op_sim"}:
        raw = config.local_prof_output_op if local else config.prof_output_op
    else:
        raw = config.local_prof_output_app if local else config.prof_output_app
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


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
        if raw not in VALID_CHIPS:
            raise ValueError(f"chip must be one of {sorted(VALID_CHIPS)}")
        return raw
    config = config or load_config()
    chip = str(config.chip or "A2").strip().upper()
    if chip not in VALID_CHIPS:
        raise ValueError(f"PERF_CHIP must be one of {sorted(VALID_CHIPS)}")
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
        "remote_env_script": config.remote_env_script if config.mode == "ssh" else None,
        "remote_path_prepend": config.remote_path_prepend if config.mode == "ssh" else None,
        "remote_conda_env": config.remote_conda_env if config.mode == "ssh" else None,
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


def build_profiler_command(payload: dict[str, Any], config: PerfRunnerConfig | None = None) -> str:
    config = config or load_config()
    prof_tool = normalize_prof_tool(payload)
    attrs = payload.get("attributes") or {}
    kernel_name = str(payload.get("kernel_name") or "").strip() or None
    warm_up = resolve_op_warm_up(payload)
    launch_count = resolve_op_launch_count(payload)
    py_args = attributes_to_cli_args(attrs, resolve_npu_device(payload, config))
    if config.mode == "ssh":
        output = config.prof_output_op if prof_tool in {"msprof_op", "msprof_op_sim"} else config.prof_output_app
    else:
        output = str(prof_output_root(prof_tool, local=True))
    remote_script, local_script = resolve_script_paths(payload, config)
    if config.mode == "ssh" and payload.get("execution_environment") is not None:
        execution = resolve_execution_environment(payload, config)
        remote_script = f"{execution.source_repo}/examples/flash_gated_delta_rule.py"
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
        return invocation
    local_output = local_prof_output_path(prof_tool, config)
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


def build_command(payload: dict[str, Any]) -> str:
    config = load_config()
    invocation = build_profiler_command(payload, config)
    if config.mode == "ssh":
        execution = resolve_execution_environment(payload, config)
        remote = _remote_execution_command(config, invocation, execution=execution)
        return " ".join(shlex.quote(part) for part in _ssh_command(config, remote))
    return invocation


def _ssh_command(config: PerfRunnerConfig, remote_command: str) -> list[str]:
    cmd = ["ssh"]
    if config.ssh_port:
        cmd.extend(["-p", config.ssh_port])
    if config.ssh_identity:
        cmd.extend(["-i", config.ssh_identity])
    cmd.extend(_ssh_connection_options())
    cmd.append(f"{config.ssh_user}@{config.ssh_host}")
    cmd.append(remote_command)
    return cmd


def _scp_command(config: PerfRunnerConfig, remote_path: str, local_path: Path) -> list[str]:
    cmd = ["scp", "-r"]
    if config.ssh_port:
        cmd.extend(["-P", config.ssh_port])
    if config.ssh_identity:
        cmd.extend(["-i", config.ssh_identity])
    cmd.extend(_ssh_connection_options())
    cmd.append(f"{config.ssh_user}@{config.ssh_host}:{remote_path}")
    cmd.append(str(local_path))
    return cmd


def _ssh_connection_options() -> list[str]:
    return [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
    ]


def _npu_metric(output: str, label: str) -> int | float | None:
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(-?\d+(?:\.\d+)?)\s*$",
        output,
        re.MULTILINE,
    )
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def parse_npu_smi_status(output: str, device_ids: list[int]) -> list[dict[str, Any]]:
    """Parse the bounded per-device output emitted by collect_npu_device_status."""
    blocks = {
        int(match.group(1)): match.group(2)
        for match in re.finditer(
            r"^__NPU__:(\d+)\r?\n(.*?)^__END_NPU__\s*$",
            output,
            re.MULTILINE | re.DOTALL,
        )
    }
    devices = []
    process_pattern = re.compile(
        r"Process id:\s*(\d+)\s+Process name:\s*(.*?)\s+Process memory\(MB\):\s*(\d+)",
    )
    for device_id in device_ids:
        block = blocks.get(device_id, "")
        usages, _, process_output = block.partition("__PROCESSES__")
        hbm_capacity = _npu_metric(usages, "HBM Capacity(MB)")
        hbm_usage = _npu_metric(usages, "HBM Usage Rate(%)")
        npu_utilization = _npu_metric(usages, "NPU Utilization(%)")
        aicore_usage = _npu_metric(usages, "Aicore Usage Rate(%)")
        aivector_usage = _npu_metric(usages, "Aivector Usage Rate(%)")
        available = any(
            value is not None
            for value in (hbm_capacity, hbm_usage, npu_utilization, aicore_usage, aivector_usage)
        )
        process_matches = list(process_pattern.finditer(process_output)) if available else []
        processes = [
            {
                "pid": int(match.group(1)),
                "name": match.group(2).strip(),
                "memory_mb": int(match.group(3)),
            }
            for match in process_matches
        ]
        process_count = len(process_matches)
        utilization_values = [
            float(value)
            for value in (npu_utilization, aicore_usage, aivector_usage)
            if value is not None
        ]
        occupied = available and (process_count > 0 or any(value > 0 for value in utilization_values))
        hbm_used = None
        if hbm_capacity is not None and hbm_usage is not None:
            hbm_used = round(float(hbm_capacity) * float(hbm_usage) / 100)
        devices.append({
            "id": device_id,
            "available": available,
            "status": "busy" if occupied else ("idle" if available else "unavailable"),
            "npu_utilization_pct": npu_utilization,
            "aicore_usage_pct": aicore_usage,
            "aivector_usage_pct": aivector_usage,
            "hbm_capacity_mb": hbm_capacity,
            "hbm_usage_pct": hbm_usage,
            "hbm_used_mb": hbm_used,
            "process_count": process_count,
            "process_memory_mb": sum(int(match.group(3)) for match in process_matches),
            "processes": processes,
            "processes_truncated": False,
        })
    return devices


def collect_npu_device_status(
    config: PerfRunnerConfig,
    *,
    device_count: int = 8,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    if config.mode != "ssh":
        raise ValueError("NPU device status currently requires PERF_RUN_MODE=ssh")
    device_ids = list(range(max(1, min(device_count, 64))))
    ids = " ".join(str(device_id) for device_id in device_ids)
    remote = (
        "export LC_ALL=C; "
        f"for device in {ids}; do "
        "printf '__NPU__:%s\\n' \"$device\"; "
        "if timeout 4s npu-smi info -t usages -i \"$device\" 2>&1; then "
        "printf '__PROCESSES__\\n'; "
        "timeout 4s npu-smi info -t proc-mem -i \"$device\" 2>&1 || true; "
        "else printf '__PROCESSES__\\n'; fi; "
        "printf '__END_NPU__\\n'; "
        "done"
    )
    try:
        result = subprocess.run(
            _ssh_command(config, remote),
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=max(10, timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"npu-smi status query timed out after {timeout_seconds}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "SSH query failed").strip()
        raise RuntimeError(f"npu-smi status query failed: {detail[:500]}")
    return parse_npu_smi_status(result.stdout, device_ids)


def _run_command(command: list[str] | str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        return subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )


def _remote_execution_command(
    config: PerfRunnerConfig,
    invocation: str,
    *,
    execution: ExecutionEnvironment | None = None,
    workdir: str | None = None,
) -> str:
    env_script = execution.env_script if execution else config.remote_env_script
    conda_env = execution.conda_env if execution else config.remote_conda_env
    if conda_env and not config.remote_conda_sh:
        raise ValueError("PERF_REMOTE_CONDA_ENV requires PERF_REMOTE_CONDA_SH")
    commands = []
    if env_script:
        commands.append(f". {shlex.quote(env_script)}")
    if config.remote_path_prepend:
        commands.append(f"export PATH={shlex.quote(config.remote_path_prepend)}:\"$PATH\"")
    if conda_env:
        commands.append(f". {shlex.quote(config.remote_conda_sh)}")
        commands.append(f"conda activate {shlex.quote(conda_env)}")
    commands.append(f"cd {shlex.quote(workdir or config.remote_workdir)}")
    commands.append(invocation)
    return " && ".join(commands)


def soc_build_target(chip: str) -> str:
    try:
        return {
            "A2": "ascend910b",
            "A3": "ascend910_93",
            "A5": "ascend950",
        }[chip.upper()]
    except KeyError as exc:
        raise ValueError(f"没有为 {chip} 配置源码构建目标") from exc


def _remote_command_error(label: str, exc: subprocess.CalledProcessError) -> RuntimeError:
    detail = (exc.stderr or exc.stdout or "远端命令执行失败").strip()
    if len(detail) > 3000:
        detail = detail[-3000:]
    return RuntimeError(f"{label}失败：{detail}")


class RemoteConnectionError(RuntimeError):
    pass


def _run_remote_checked(
    config: PerfRunnerConfig,
    remote_command: str,
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return _run_command(_ssh_command(config, remote_command))
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 255:
            detail = (exc.stderr or exc.stdout or "SSH connection failed").strip()
            raise RemoteConnectionError(f"{label}连接失败：{detail[-1000:]}") from None
        raise _remote_command_error(label, exc) from None


def _validate_remote_execution_environment(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
) -> None:
    checks = [
        f"test -f {shlex.quote(execution.env_script)}",
        f"test -f {shlex.quote(config.remote_conda_sh)}",
        f"test -d {shlex.quote(execution.source_repo)}/.git",
        "python --version",
    ]
    if not execution.rebuild:
        checks.append(f"test -f {shlex.quote(execution.source_repo)}/examples/flash_gated_delta_rule.py")
    remote = _remote_execution_command(
        config,
        " && ".join(checks),
        execution=execution,
        workdir=execution.source_repo,
    )
    _run_remote_checked(config, remote, "执行环境检查")


def _remote_build_worktree_path(config: PerfRunnerConfig) -> str:
    root = _normalized_remote_absolute_path(config.remote_build_root, "远端构建目录")
    return f"{root}/{uuid.uuid4().hex}"


def _remote_deployment_path(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
    chip: str,
) -> str:
    identity = json.dumps({
        "cann_path": execution.cann_path,
        "conda_env": execution.conda_env,
        "source_repo": execution.source_repo,
        "chip": chip.upper(),
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    deployment_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    root = _normalized_remote_absolute_path(config.remote_build_root, "远端构建目录")
    return f"{root}/active/{deployment_id}"


def _prepare_remote_source_build(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
    chip: str,
    *,
    worktree: str | None = None,
) -> dict[str, str]:
    worktree = worktree or _remote_build_worktree_path(config)
    repo = shlex.quote(execution.source_repo)
    branch_ref = shlex.quote(f"refs/heads/{execution.branch}^{{commit}}")
    origin_ref = shlex.quote(f"refs/remotes/origin/{execution.branch}^{{commit}}")
    worktree_arg = shlex.quote(worktree)
    if execution.branch_source == "remote":
        resolve_commit = (
            f"commit=$(git -C {repo} rev-parse --verify {origin_ref} 2>/dev/null || true)"
            " && (export GIT_TERMINAL_PROMPT=0; "
            f"timeout --signal=TERM --kill-after=2s 30s git -C {repo} "
            "fetch --prune origin >/dev/null 2>&1 || true)"
            f" && updated=$(git -C {repo} rev-parse --verify {origin_ref} 2>/dev/null || true)"
            " && if [ -n \"$updated\" ]; then commit=\"$updated\"; fi"
        )
    else:
        resolve_commit = f"commit=$(git -C {repo} rev-parse --verify {branch_ref} 2>/dev/null || true)"
    prepare = (
        f"git -C {repo} check-ref-format --branch {shlex.quote(execution.branch)}"
        f" && {resolve_commit}"
        f" && test -n \"$commit\""
        f" && mkdir -p {shlex.quote(str(PurePosixPath(worktree).parent))}"
        f" && git -C {repo} worktree add --detach {worktree_arg} \"$commit\""
        " && printf '%s\\n' \"$commit\""
    )
    try:
        prepared = _run_remote_checked(config, prepare, "源码分支准备")
        commit = next((line.strip() for line in reversed(prepared.stdout.splitlines()) if line.strip()), "")
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
            raise RuntimeError("源码分支准备失败：未能确定提交版本")
        build = (
            "python scripts/check_npu_env.py --build-only"
            " && mkdir -p dist"
            f" && FLA_NPU_SOC={shlex.quote(soc_build_target(chip))} "
            "python -m pip wheel --no-build-isolation --no-deps . -w dist"
            " && wheel=$(find dist -maxdepth 1 -type f "
            "-name 'flash_linear_attention_npu-*.whl' -print | sort | tail -n 1)"
            " && test -n \"$wheel\""
            " && python -m pip install --force-reinstall --no-cache-dir --no-deps \"$wheel\""
        )
        remote_build = _remote_execution_command(
            config,
            build,
            execution=execution,
            workdir=worktree,
        )
        _run_remote_checked(config, remote_build, "源码编译安装")
        return {
            "branch": execution.branch,
            "branch_source": execution.branch_source,
            "commit": commit,
            "soc": soc_build_target(chip),
            "worktree": worktree,
        }
    except Exception:
        _cleanup_remote_source_build(config, execution.source_repo, worktree)
        raise


def persistent_build_handle(
    payload: dict[str, Any],
    execution_id: str,
    config: PerfRunnerConfig | None = None,
) -> dict[str, Any]:
    config = config or ensure_runner_configured()
    if config.mode != "ssh":
        raise RuntimeError("编译安装任务仅支持 SSH Relay 模式")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", execution_id):
        raise ValueError("invalid persistent build execution id")
    execution = resolve_execution_environment(payload, config)
    if not execution.branch:
        raise ValueError("编译安装任务必须指定源码分支")
    chip = resolve_chip(payload, config)
    worktree = (
        PurePosixPath(_normalized_remote_absolute_path(config.remote_build_root, "远端构建目录"))
        / "attempts"
        / execution_id
    ).as_posix()
    return {
        "execution_id": execution_id,
        "worktree": worktree,
        "control_dir": f"{worktree}/.fla-runner",
        "log_path": f"{worktree}/.fla-runner/build.log",
        "source_repo": execution.source_repo,
        "branch": execution.branch,
        "branch_source": execution.branch_source,
        "chip": chip,
        "soc": soc_build_target(chip),
        "execution_environment": execution_environment_summary(execution),
    }


def _persistent_build_script(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
    chip: str,
    build_info: dict[str, str],
) -> str:
    worktree = build_info["worktree"]
    control = f"{worktree}/.fla-runner"
    deployment = _remote_deployment_path(config, execution, chip)
    marker = f"{worktree}/.fla-runner-deployment"
    build_root = _normalized_remote_absolute_path(config.remote_build_root, "远端构建目录")
    setup = _remote_execution_command(
        config,
        (
            "python scripts/check_npu_env.py --build-only"
            " && mkdir -p dist"
            f" && FLA_NPU_SOC={shlex.quote(soc_build_target(chip))} "
            "python -m pip wheel --no-build-isolation --no-deps . -w dist"
            " && wheel=$(find dist -maxdepth 1 -type f "
            "-name 'flash_linear_attention_npu-*.whl' -print | sort | tail -n 1)"
            " && test -n \"$wheel\""
            " && python -m pip install --force-reinstall --no-cache-dir --no-deps \"$wheel\""
        ),
        execution=execution,
        workdir=worktree,
    )
    metadata = " ".join(
        shlex.quote(value)
        for value in (
            execution.branch,
            execution.branch_source,
            build_info["commit"],
            build_info["soc"],
        )
    )
    return f"""#!/usr/bin/env bash
set +e
control={shlex.quote(control)}
state_file="$control/state"
exit_file="$control/exit_code"
log_file="$control/build.log"
write_value() {{
  printf '%s\n' "$2" > "$1.tmp"
  mv -f "$1.tmp" "$1"
}}
finish() {{
  write_value "$exit_file" "$1"
  write_value "$state_file" "$2"
}}
cancel_build() {{
  printf '%s\n' '[runner] cancellation signal received' >> "$log_file"
  finish 143 canceled
  exit 143
}}
trap cancel_build TERM INT
mkdir -p "$control"
printf '%s\n' "$$" > "$control/pid"
write_value "$state_file" running
rm -f "$exit_file"
printf '%s\n' '[runner] persistent build started' >> "$log_file"
(
  set -e
  {setup}
  mkdir -p {shlex.quote(str(PurePosixPath(deployment).parent))}
  printf '%s\n' {metadata} > {shlex.quote(marker)}
  previous=$(readlink -f {shlex.quote(deployment)} 2>/dev/null || true)
  ln -sfn {shlex.quote(worktree)} {shlex.quote(deployment)}
  if [ -n "$previous" ] && [ "$previous" != {shlex.quote(worktree)} ]; then
    case "$previous" in
      {shlex.quote(build_root)}/*)
        git -C {shlex.quote(execution.source_repo)} worktree remove --force "$previous" >/dev/null 2>&1 || true
        git -C {shlex.quote(execution.source_repo)} worktree prune >/dev/null 2>&1 || true
        ;;
    esac
  fi
) >> "$log_file" 2>&1
code=$?
if [ "$code" -eq 0 ]; then
  printf '%s\n' '[runner] persistent build completed' >> "$log_file"
  finish 0 succeeded
else
  printf '[runner] persistent build failed with exit code %s\n' "$code" >> "$log_file"
  finish "$code" failed
fi
exit "$code"
"""


def start_persistent_build_install(
    payload: dict[str, Any],
    handle: dict[str, Any],
    config: PerfRunnerConfig | None = None,
) -> dict[str, Any]:
    config = config or ensure_runner_configured()
    execution = resolve_execution_environment(payload, config)
    chip = resolve_chip(payload, config)
    _validate_remote_execution_environment(config, execution)
    status = poll_persistent_build_install(handle, config, allow_missing=True)
    if status["state"] in {"running", "succeeded", "failed", "canceled"}:
        return {**handle, **{key: status.get(key) for key in ("commit", "soc", "deployment") if status.get(key)}}
    worktree = str(handle["worktree"])
    _cleanup_remote_source_build(config, execution.source_repo, worktree)
    build_info = _prepare_remote_source_build(config, execution, chip, worktree=worktree)
    script = _persistent_build_script(config, execution, chip, build_info)
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    control = shlex.quote(str(handle["control_dir"]))
    script_path = shlex.quote(f"{handle['control_dir']}/run-build.sh")
    remote = (
        f"mkdir -p {control}"
        f" && printf '%s' {shlex.quote(encoded)} | base64 -d > {script_path}"
        f" && chmod 700 {script_path}"
        f" && {{ setsid nohup {script_path} </dev/null >/dev/null 2>&1 &"
        f" starter=$!; printf '%s\n' \"$starter\"; }}"
    )
    _run_remote_checked(config, remote, "持久编译任务启动")
    return {
        **handle,
        "commit": build_info["commit"],
        "soc": build_info["soc"],
        "deployment": _remote_deployment_path(config, execution, chip),
    }


def _parse_persistent_build_status(output: str, handle: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith("__"):
            values[key.strip("_")] = value
    log_tail = ""
    encoded_log = values.get("LOG_BASE64", "")
    if encoded_log:
        try:
            log_tail = base64.b64decode(encoded_log, validate=True).decode("utf-8", errors="replace")
        except (ValueError, binascii.Error):
            log_tail = "[runner] build log could not be decoded"
    raw_exit = values.get("EXIT_CODE", "")
    try:
        exit_code = int(raw_exit) if raw_exit else None
    except ValueError:
        exit_code = None
    state = values.get("STATE", "missing") or "missing"
    alive = values.get("ALIVE") == "1"
    if state == "running" and not alive:
        state = "failed"
        if exit_code is None:
            exit_code = -1
        log_tail = (log_tail + "\n[runner] remote build process disappeared without a final state").strip()
    return {
        **handle,
        "state": state,
        "pid": values.get("PID", ""),
        "alive": alive,
        "exit_code": exit_code,
        "log_tail": log_tail,
        "log_size": int(values.get("LOG_SIZE", "0") or 0),
    }


def poll_persistent_build_install(
    handle: dict[str, Any],
    config: PerfRunnerConfig | None = None,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    config = config or ensure_runner_configured()
    control = shlex.quote(str(handle["control_dir"]))
    remote = (
        f"control={control}; "
        "state=$(cat \"$control/state\" 2>/dev/null || true); "
        "pid=$(cat \"$control/pid\" 2>/dev/null || true); "
        "exit_code=$(cat \"$control/exit_code\" 2>/dev/null || true); "
        "alive=0; if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then alive=1; fi; "
        "log_size=$(wc -c < \"$control/build.log\" 2>/dev/null || printf 0); "
        "log_base64=$(tail -c 16000 \"$control/build.log\" 2>/dev/null | base64 | tr -d '\\n'); "
        "printf '__STATE__=%s\\n__PID__=%s\\n__ALIVE__=%s\\n__EXIT_CODE__=%s\\n__LOG_SIZE__=%s\\n__LOG_BASE64__=%s\\n' "
        "\"${state:-missing}\" \"$pid\" \"$alive\" \"$exit_code\" \"$log_size\" \"$log_base64\""
    )
    result = _run_remote_checked(config, remote, "持久编译状态查询")
    return _parse_persistent_build_status(result.stdout, handle)


def cancel_persistent_build_install(
    handle: dict[str, Any],
    config: PerfRunnerConfig | None = None,
) -> None:
    config = config or ensure_runner_configured()
    control = shlex.quote(str(handle["control_dir"]))
    remote = (
        f"control={control}; pid=$(cat \"$control/pid\" 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then "
        "kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\" 2>/dev/null || true; fi"
    )
    _run_remote_checked(config, remote, "持久编译任务取消")


def persistent_build_result(handle: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    if status.get("state") != "succeeded":
        detail = str(status.get("log_tail") or "远端编译未返回日志").strip()
        raise RuntimeError(f"源码编译安装失败（退出码 {status.get('exit_code')}）：{detail[-3000:]}")
    environment = dict(handle.get("execution_environment") or {})
    environment.update({
        "commit": handle.get("commit"),
        "soc": handle.get("soc"),
        "deployment": handle.get("deployment"),
        "build_log": handle.get("log_path"),
    })
    return {
        "status": "done",
        "task_type": "build_install",
        "message": f"源码分支 {handle.get('branch_source')}:{handle.get('branch')} 编译安装完成",
        "execution_environment": environment,
        "build_log": handle.get("log_path"),
        "build_log_tail": status.get("log_tail") or "",
    }


def _cleanup_remote_source_build(config: PerfRunnerConfig, source_repo: str, worktree: str) -> None:
    if not worktree:
        return
    remote = (
        f"git -C {shlex.quote(source_repo)} worktree remove --force {shlex.quote(worktree)} >/dev/null 2>&1"
        f" || rm -rf {shlex.quote(worktree)}; "
        f"git -C {shlex.quote(source_repo)} worktree prune >/dev/null 2>&1 || true"
    )
    try:
        _run_command(_ssh_command(config, remote))
    except Exception:
        pass


def _activate_remote_source_build(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
    chip: str,
    build_info: dict[str, str],
) -> str:
    worktree = build_info["worktree"]
    deployment = _remote_deployment_path(config, execution, chip)
    marker = f"{worktree}/.fla-runner-deployment"
    active_root = str(PurePosixPath(deployment).parent)
    command = (
        f"mkdir -p {shlex.quote(active_root)}"
        f" && printf '%s\\n' {shlex.quote(execution.branch)} {shlex.quote(execution.branch_source)} "
        f"{shlex.quote(build_info['commit'])} "
        f"{shlex.quote(build_info['soc'])} > {shlex.quote(marker)}"
        f" && previous=$(readlink -f {shlex.quote(deployment)} 2>/dev/null || true)"
        f" && ln -sfn {shlex.quote(worktree)} {shlex.quote(deployment)}"
        " && printf '%s\\n' \"$previous\""
    )
    activated = _run_remote_checked(config, command, "构建版本激活")
    previous = next((line.strip() for line in activated.stdout.splitlines() if line.strip()), "")
    build_root = _normalized_remote_absolute_path(config.remote_build_root, "远端构建目录")
    if previous and previous != worktree and _path_allowed(previous, (build_root,)):
        _cleanup_remote_source_build(config, execution.source_repo, previous)
    return deployment


def _resolve_remote_deployed_source(
    config: PerfRunnerConfig,
    execution: ExecutionEnvironment,
    chip: str,
) -> str:
    deployment = _remote_deployment_path(config, execution, chip)
    marker = f"{deployment}/.fla-runner-deployment"
    inspect = (
        f"test -L {shlex.quote(deployment)}"
        f" && test -f {shlex.quote(marker)}"
        f" && cat {shlex.quote(marker)}"
    )
    try:
        result = _run_remote_checked(config, inspect, "已安装源码版本检查")
    except RuntimeError:
        raise RuntimeError(
            f"源码分支 {execution.branch_source}:{execution.branch} 尚未在当前 CANN/Conda 环境编译安装，"
            "请先执行编译安装任务"
        ) from None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    expected_soc = soc_build_target(chip)
    if len(lines) >= 4:
        installed_branch, installed_source, _, installed_soc = lines[:4]
    else:
        installed_branch, installed_source = (lines[0] if lines else ""), "local"
        installed_soc = lines[2] if len(lines) >= 3 else ""
    if (
        installed_branch != execution.branch
        or installed_source != execution.branch_source
        or installed_soc != expected_soc
    ):
        installed = lines[0] if lines else "未知"
        raise RuntimeError(
            f"当前环境安装的是源码分支 {installed_source}:{installed}，"
            f"不是 {execution.branch_source}:{execution.branch}，请先执行编译安装任务"
        )
    return deployment


def _remote_output_path(config: PerfRunnerConfig, output: str) -> str:
    path = PurePosixPath(output)
    if path.is_absolute():
        return path.as_posix()
    return (PurePosixPath(config.remote_workdir) / path).as_posix()


def _list_remote_prof_dirs(config: PerfRunnerConfig, prof_tool: str) -> set[str]:
    output = config.prof_output_op if prof_tool in {"msprof_op", "msprof_op_sim"} else config.prof_output_app
    prefix = prof_dir_prefix(prof_tool)
    remote_output = _remote_output_path(config, output).rstrip("/")
    remote = f"ls -1d {shlex.quote(remote_output)}/{prefix}* 2>/dev/null || true"
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


def _execute_build_install(payload: dict[str, Any], config: PerfRunnerConfig) -> dict[str, Any]:
    if config.mode != "ssh":
        raise RuntimeError("编译安装任务仅支持 SSH Relay 模式")
    execution = resolve_execution_environment(payload, config)
    if not execution.branch:
        raise ValueError("编译安装任务必须指定源码分支")
    chip = resolve_chip(payload, config)
    if config.dry_run:
        return {
            "status": "done",
            "task_type": "build_install",
            "message": "dry-run：未实际执行编译安装",
            "execution_environment": execution_environment_summary(execution),
            "dry_run": True,
        }

    _validate_remote_execution_environment(config, execution)
    build_info = _prepare_remote_source_build(config, execution, chip)
    activated = False
    try:
        deployment = _activate_remote_source_build(config, execution, chip, build_info)
        activated = True
    finally:
        if not activated:
            _cleanup_remote_source_build(config, execution.source_repo, build_info["worktree"])
    return {
        "status": "done",
        "task_type": "build_install",
        "message": f"源码分支 {execution.branch_source}:{execution.branch} 编译安装完成",
        "execution_environment": {
            **execution_environment_summary(execution),
            "commit": build_info["commit"],
            "soc": build_info["soc"],
            "deployment": deployment,
        },
    }


def execute(payload: dict[str, Any], *, persist_local_data: bool = True) -> dict[str, Any]:
    config = ensure_runner_configured()
    task_type = normalize_task_type(payload)
    if task_type == "build_install":
        return _execute_build_install(payload, config)
    prof_tool = normalize_prof_tool(payload)
    chip = resolve_chip(payload, config)
    execution = resolve_execution_environment(payload, config) if config.mode == "ssh" else None
    command = build_command(payload)
    profiler_command = build_profiler_command(payload, config)
    if config.dry_run:
        return {
            "status": "done",
            "message": "dry-run：未实际执行",
            "command": command,
            "profiler_command": profiler_command,
            "prof_tool": prof_tool,
            "dry_run": True,
            "execution_environment": execution_environment_summary(execution) if execution else {},
        }

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
        assert execution is not None
        build_info: dict[str, str] = {}
        build_worktree = ""
        if execution.customized:
            _validate_remote_execution_environment(config, execution)
            remote_script = f"{execution.source_repo}/examples/flash_gated_delta_rule.py"
            if execution.branch and not execution.rebuild:
                deployed_source = _resolve_remote_deployed_source(config, execution, chip)
                remote_script = f"{deployed_source}/examples/flash_gated_delta_rule.py"
        try:
            if execution.rebuild:
                build_info = _prepare_remote_source_build(config, execution, chip)
                build_worktree = build_info["worktree"]
                remote_script = f"{build_worktree}/examples/flash_gated_delta_rule.py"
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
            profiler_command = invocation
            remote = _remote_execution_command(config, invocation, execution=execution)
            command = " ".join(shlex.quote(part) for part in _ssh_command(config, remote))
            _run_remote_checked(config, remote, prof_tool_label(prof_tool))
            after = _list_remote_prof_dirs(config, prof_tool)
        finally:
            if build_worktree:
                _cleanup_remote_source_build(config, execution.source_repo, build_worktree)
        prof_name = _resolve_new_prof_dir(before, after, prof_tool)
        local_dir = local_root / prof_name
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        remote_prof = f"{_remote_output_path(config, remote_output).rstrip('/')}/{prof_name}"
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
                output=local_prof_output_path(prof_tool, config),
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
        data = import_module.import_prof(
            prof_dir,
            model_id,
            chip,
            replace_mock=False,
            device_id=npu_device,
            persist=persist_local_data,
        )
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
            persist=persist_local_data,
        )
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)

    data["runs"] = [
        item
        for item in data.get("runs", [])
        if not (item.get("created_by") in {"import_prof_gdr", "import_msprof_op"} and item.get("snapshot_id") == snapshot["id"])
    ]
    if persist_local_data:
        for path in import_module.PERF_PATHS:
            path.write_text(
                __import__("json").dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return {
        "status": "done",
        "message": f"{prof_tool_label(prof_tool)} 执行并导入：{prof_dir.name}",
        "command": command,
        "profiler_command": profiler_command,
        "prof_tool": prof_tool,
        "prof_dir": str(prof_dir),
        "prof_source": prof_dir.name,
        "execution_environment": {
            **execution_environment_summary(execution),
            **({"commit": build_info.get("commit"), "soc": build_info.get("soc")} if build_info else {}),
        } if execution else {},
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
