from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "perf-examples.json"
DEFAULT_EXAMPLE_ID = "flash_gated_delta_rule"
ATTRIBUTE_METADATA_KEYS = {"layout", "notes"}


def load_example_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or MANIFEST_PATH
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("examples"), list):
        raise ValueError("性能示例 manifest 格式不合法")
    ids: set[str] = set()
    aliases: set[str] = set()
    for example in value["examples"]:
        example_id = str(example.get("id") or "")
        script = str(example.get("script") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", example_id):
            raise ValueError(f"性能示例 ID 不合法：{example_id}")
        if example_id in ids or not re.fullmatch(r"examples/[A-Za-z0-9_./-]+\.py", script):
            raise ValueError(f"性能示例定义不合法：{example_id}")
        ids.add(example_id)
        aliases.update(str(item) for item in example.get("legacy_ids", []))
        parameter_names: set[str] = set()
        for parameter in example.get("parameters", []):
            name = str(parameter.get("name") or "")
            flag = str(parameter.get("flag") or "")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in parameter_names:
                raise ValueError(f"示例 {example_id} 的参数名称不合法：{name}")
            if not re.fullmatch(r"--[a-z][a-z0-9-]{0,63}", flag):
                raise ValueError(f"示例 {example_id} 的参数 flag 不合法：{flag}")
            parameter_names.add(name)
    if ids & aliases:
        raise ValueError("性能示例 ID 与兼容别名冲突")
    return value


EXAMPLE_MANIFEST = load_example_manifest()


def example_catalog() -> list[dict[str, Any]]:
    return copy.deepcopy(EXAMPLE_MANIFEST["examples"])


def example_schema_version() -> int:
    return int(EXAMPLE_MANIFEST["schema_version"])


def example_manifest_hash() -> str:
    encoded = json.dumps(EXAMPLE_MANIFEST, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_example(value: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(value, dict):
        requested_version = value.get("example_schema_version")
        if requested_version is not None and int(requested_version) != example_schema_version():
            raise ValueError(
                f"不支持测试示例 schema 版本：{requested_version}，当前版本为 {example_schema_version()}"
            )
        requested = str(
            value.get("example_id")
            or value.get("script_id")
            or value.get("script_path")
            or DEFAULT_EXAMPLE_ID
        ).strip()
    else:
        requested = str(value or DEFAULT_EXAMPLE_ID).strip()
    for example in EXAMPLE_MANIFEST["examples"]:
        if requested == example["id"] or requested in example.get("legacy_ids", []):
            return example
    raise ValueError(f"未知测试示例：{requested}")


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} 必须为布尔值")


def _number_value(value: Any, parameter: dict[str, Any], *, integer: bool) -> int | float | None:
    name = parameter["name"]
    if value in (None, "") and parameter.get("default") is None:
        return None
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为{'整数' if integer else '数字'}") from exc
    if not math.isfinite(float(number)):
        raise ValueError(f"{name} 必须为有限数字")
    if "min" in parameter and number < parameter["min"]:
        raise ValueError(f"{name} 不能小于 {parameter['min']}")
    if "max" in parameter and number > parameter["max"]:
        raise ValueError(f"{name} 不能大于 {parameter['max']}")
    if "min_exclusive" in parameter and number <= parameter["min_exclusive"]:
        raise ValueError(f"{name} 必须大于 {parameter['min_exclusive']}")
    if "max_exclusive" in parameter and number >= parameter["max_exclusive"]:
        raise ValueError(f"{name} 必须小于 {parameter['max_exclusive']}")
    step = parameter.get("step")
    if step and integer and number % int(step) != 0:
        raise ValueError(f"{name} 必须为 {step} 的整数倍")
    return number


def _integer_list(value: Any, parameter: dict[str, Any]) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[\s,]+", value.strip()) if value.strip() else []
    elif isinstance(value, list):
        raw_items = value
    else:
        raise ValueError(f"{parameter['name']} 必须为整数列表")
    if len(raw_items) > 4096:
        raise ValueError(f"{parameter['name']} 最多包含 4096 个整数")
    try:
        return [int(item) for item in raw_items]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter['name']} 必须为整数列表") from exc


def _normalize_parameter(value: Any, parameter: dict[str, Any]) -> Any:
    parameter_type = parameter["type"]
    name = parameter["name"]
    if parameter_type == "boolean":
        return _bool_value(value, name)
    if parameter_type == "integer":
        return _number_value(value, parameter, integer=True)
    if parameter_type == "number":
        return _number_value(value, parameter, integer=False)
    if parameter_type == "integer_list":
        return _integer_list(value, parameter)
    if parameter_type == "enum":
        choices = parameter.get("choices", [])
        if value not in choices and str(value) not in {str(choice) for choice in choices}:
            raise ValueError(f"{name} 必须为 {choices} 之一")
        return next(choice for choice in choices if str(choice) == str(value))
    if parameter_type == "string":
        text = str(value or "")
        if len(text) > int(parameter.get("max_length", 500)) or any(ord(char) < 32 for char in text):
            raise ValueError(f"{name} 包含不支持的字符或长度超限")
        return text
    raise ValueError(f"{name} 使用了不支持的参数类型：{parameter_type}")


def normalize_example_attributes(example: dict[str, Any], attributes: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(attributes or {})
    parameters = {item["name"]: item for item in example.get("parameters", [])}
    unknown = set(raw) - set(parameters) - ATTRIBUTE_METADATA_KEYS
    if unknown:
        raise ValueError(f"示例 {example['id']} 不支持参数：{', '.join(sorted(unknown))}")
    normalized: dict[str, Any] = {}
    for name, parameter in parameters.items():
        value = raw[name] if name in raw else copy.deepcopy(parameter.get("default"))
        normalized[name] = _normalize_parameter(value, parameter)
    _validate_cross_fields(example["id"], normalized)
    normalized["layout"] = "TND" if normalized.get("varlen") else "BSND"
    if raw.get("notes"):
        normalized["notes"] = str(raw["notes"])[:500]
    return normalized


def _validate_cross_fields(example_id: str, attrs: dict[str, Any]) -> None:
    query_heads = int(attrs.get("query_heads") or 1)
    value_heads = int(attrs.get("value_heads") or query_heads)
    if example_id == "flash_kda":
        if query_heads != value_heads:
            raise ValueError("Flash KDA 要求 query-heads 等于 value-heads")
        if attrs.get("varlen") and int(attrs.get("batch") or 1) != 1:
            raise ValueError("Flash KDA 变长输入要求 batch=1")
    if example_id == "flash_gated_delta_rule" and attrs.get("demo_model") and int(attrs.get("batch") or 1) != 1:
        raise ValueError("--demo-model requires batch=1")
    if example_id in {"recurrent_gated_delta_rule", "recurrent_kda_layer"}:
        if value_heads % query_heads != 0:
            raise ValueError("Recurrent 示例要求 value-heads 可被 query-heads 整除")
        if int(attrs.get("mtp") or 1) > 1 and attrs.get("use_short_conv", True) and int(attrs.get("conv_kernel") or 4) != 4:
            raise ValueError("MTP 大于 1 且启用短卷积时 conv-kernel 必须为 4")
    if example_id == "recurrent_kda_layer" and int(attrs.get("key_dim") or 0) != 128:
        raise ValueError("Recurrent KDA 要求 key-dim=128")
    if attrs.get("safe_gate") and not -5.0 <= float(attrs.get("lower_bound") or -5.0) < 0.0:
        raise ValueError("safe gate 要求 lower-bound 位于 [-5, 0)")


def example_cli_args(example: dict[str, Any], attributes: dict[str, Any], npu_device: int) -> list[str]:
    normalized = normalize_example_attributes(example, attributes)
    args = ["--device", str(npu_device)]
    for parameter in example.get("parameters", []):
        name = parameter["name"]
        value = normalized[name]
        if parameter.get("visible_when"):
            if any(normalized.get(key) != expected for key, expected in parameter["visible_when"].items()):
                continue
        if parameter["type"] == "boolean":
            if value:
                args.append(parameter["flag"])
            elif parameter.get("false_flag"):
                args.append(parameter["false_flag"])
            continue
        if value is None or value == "" or value == []:
            continue
        args.append(parameter["flag"])
        if parameter["type"] == "integer_list":
            if parameter.get("separator") == ",":
                args.append(",".join(str(item) for item in value))
            else:
                args.extend(str(item) for item in value)
        else:
            args.append(str(value))
    return args
