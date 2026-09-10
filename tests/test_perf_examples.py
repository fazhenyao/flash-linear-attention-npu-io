import copy
import json
import tempfile
import unittest
from pathlib import Path

from backend.perf_examples import (
    EXAMPLE_MANIFEST,
    example_catalog,
    example_cli_args,
    load_example_manifest,
    normalize_example_attributes,
    resolve_example,
)


class PerfExampleManifestTests(unittest.TestCase):
    def test_all_example_scripts_are_stored_in_repository(self):
        root = Path(__file__).resolve().parents[1]
        for example in example_catalog():
            relative = example.get("local_script") or example["script"]
            script = (root / relative).resolve()
            self.assertTrue(script.is_relative_to(root), relative)
            self.assertTrue(script.is_file(), relative)

    def test_manifest_rejects_example_script_missing_from_repository(self):
        manifest = copy.deepcopy(EXAMPLE_MANIFEST)
        manifest["examples"][0]["local_script"] = "examples/not_in_repository.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "perf-examples.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "本仓脚本不存在"):
                load_example_manifest(path)

    def test_catalog_contains_all_supported_examples(self):
        self.assertEqual(
            [item["id"] for item in example_catalog()],
            [
                "flash_gated_delta_rule",
                "flash_kda",
                "recurrent_gated_delta_rule",
                "recurrent_kda_layer",
            ],
        )

    def test_legacy_gdr_script_id_resolves_to_manifest_example(self):
        example = resolve_example("scripts/flash_gated_delta_rule.py")
        self.assertEqual(example["id"], "flash_gated_delta_rule")

    def test_rejects_unknown_manifest_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            resolve_example({"example_id": "flash_kda", "example_schema_version": 2})

    def test_flash_kda_defaults_match_repository_script(self):
        example = resolve_example("flash_kda")
        attributes = normalize_example_attributes(example, {})
        args = example_cli_args(example, attributes, 7)

        self.assertEqual(args[:2], ["--device", "7"])
        self.assertEqual(attributes["tokens"], 65536)
        self.assertEqual(attributes["query_heads"], 32)
        self.assertEqual(attributes["value_heads"], 32)
        self.assertEqual(attributes["mean_len"], 1024)
        self.assertIn("--chunk-size", args)
        self.assertIn("--qk-l2norm", args)
        self.assertIn("--varlen", args)
        self.assertNotIn("--hidden-size", args)
        self.assertNotIn("--use-short-conv", args)

    def test_flash_kda_dense_and_no_l2norm_build_false_flags(self):
        example = resolve_example("flash_kda")
        attributes = normalize_example_attributes(example, {"varlen": False, "qk_l2norm": False})
        args = example_cli_args(example, attributes, 7)

        self.assertIn("--no-varlen", args)
        self.assertIn("--no-qk-l2norm", args)

    def test_flash_gdr_composite_core_builds_true_and_false_flags(self):
        example = resolve_example("flash_gated_delta_rule")
        parameter = next(item for item in example["parameters"] if item["name"] == "use_composite_core")

        self.assertEqual(parameter["chips"], ["A2", "A5"])
        enabled = normalize_example_attributes(example, {})
        enabled_args = example_cli_args(example, enabled, 2)
        self.assertTrue(enabled["use_composite_core"])
        self.assertNotIn("--use-composite-core", enabled_args)
        self.assertNotIn("--legacy-unfused-core", enabled_args)

        disabled = normalize_example_attributes(example, {"use_composite_core": False})
        disabled_args = example_cli_args(example, disabled, 2)
        self.assertFalse(disabled["use_composite_core"])
        self.assertNotIn("--use-composite-core", disabled_args)
        self.assertIn("--legacy-unfused-core", disabled_args)

    def test_recurrent_integer_lists_are_forwarded_as_nargs(self):
        example = resolve_example("recurrent_gated_delta_rule")
        attributes = normalize_example_attributes(example, {"cache_indices": [0, 3, 7]})
        args = example_cli_args(example, attributes, 0)
        start = args.index("--cache-indices")

        self.assertEqual(args[start + 1:start + 4], ["0", "3", "7"])

    def test_cross_field_constraints_are_rejected(self):
        flash_kda = resolve_example("flash_kda")
        with self.assertRaisesRegex(ValueError, "query-heads"):
            normalize_example_attributes(flash_kda, {"query_heads": 2, "value_heads": 4})
        with self.assertRaisesRegex(ValueError, "Q/K L2Norm"):
            normalize_example_attributes(flash_kda, {"demo_model": True, "qk_l2norm": False})

        recurrent = resolve_example("recurrent_kda_layer")
        with self.assertRaisesRegex(ValueError, "conv-kernel"):
            normalize_example_attributes(recurrent, {"mtp": 2, "conv_kernel": 3})


if __name__ == "__main__":
    unittest.main()
