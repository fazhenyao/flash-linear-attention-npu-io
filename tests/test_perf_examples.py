import unittest

from backend.perf_examples import (
    example_catalog,
    example_cli_args,
    normalize_example_attributes,
    resolve_example,
)


class PerfExampleManifestTests(unittest.TestCase):
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

    def test_flash_kda_defaults_build_expected_flags(self):
        example = resolve_example("flash_kda")
        attributes = normalize_example_attributes(example, {})
        args = example_cli_args(example, attributes, 7)

        self.assertEqual(args[:2], ["--device", "7"])
        self.assertIn("--chunk-size", args)
        self.assertIn("--use-short-conv", args)
        self.assertNotIn("--varlen", args)

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

        recurrent = resolve_example("recurrent_kda_layer")
        with self.assertRaisesRegex(ValueError, "conv-kernel"):
            normalize_example_attributes(recurrent, {"mtp": 2, "conv_kernel": 3})


if __name__ == "__main__":
    unittest.main()
