import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import import_prof_gdr as importer


class OpSummaryDetailsTests(unittest.TestCase):
    def test_exact_names_unmapped_ops_and_sub_microsecond_precision_survive(self):
        rows = [
            {"Op Name": "kernel_variant_1", "OP Type": "L2Norm", "Task Duration(us)": "0.125", "aic_mac_time(us)": "2", "Task Start Time(us)": "100"},
            {"Op Name": "kernel_variant_1", "OP Type": "L2Norm", "Task Duration(us)": "0.375", "aic_mac_time(us)": "N/A", "Task Start Time(us)": "200"},
            {"OP_Name": "kernel_variant_2", "OP Type": "L2Norm", "Task Duration(us)": "4", "Task Wait Time(us)": "0"},
            {"Op Name": "unmapped_op", "OP Type": "Unknown", "custom_time(ms)": "1.5", "aic_mac_ratio": "0.5"},
        ]
        entries = {item["op_name"]: item for item in importer.summarize_named_ops(rows)}
        self.assertEqual(set(entries), {"kernel_variant_1", "kernel_variant_2", "unmapped_op"})
        self.assertEqual(entries["kernel_variant_1"]["call_count"], 2)
        stats = entries["kernel_variant_1"]["time_stats"]
        self.assertEqual(stats["Task Duration(us)"], {"count": 2, "total": 0.5, "avg": 0.25, "min": 0.125, "max": 0.375})
        self.assertEqual(stats["aic_mac_time(us)"]["avg"], 2)
        self.assertEqual(stats["aic_mac_time(us)"]["count"], 1)
        self.assertEqual(stats["Task Start Time(us)"], {"count": 2, "min": 100, "max": 200})
        self.assertEqual(entries["kernel_variant_2"]["time_stats"]["Task Wait Time(us)"]["total"], 0)
        self.assertEqual(set(entries["unmapped_op"]["time_stats"]), {"custom_time(ms)"})

    def test_missing_and_non_finite_measurements_are_not_zero(self):
        rows = [{"Op Name": "x", "Task Duration(us)": value} for value in ["", "N/A", "NaN", "inf"]]
        self.assertEqual(importer.summarize_named_ops(rows), [{"op_name": "x", "call_count": 4, "time_stats": {}}])

    def test_explicit_block_categories_do_not_merge_identical_names(self):
        rows = [
            {"Op Name": "x", "Block": "Prefill", "SubBlock": "Attention", "Task Duration(us)": "1"},
            {"Op Name": "x", "Block": "Prefill", "SubBlock": "MLP", "Task Duration(us)": "2"},
            {"Op Name": "x", "Task Duration(us)": "3"},
        ]
        result = importer.summarize_named_ops(rows)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["sub_block"], "Attention")
        self.assertEqual(result[1]["time_stats"]["Task Duration(us)"]["total"], 2)
        self.assertNotIn("block", result[2])

    def test_import_preserves_details_in_result_snapshot(self):
        raw_rows = [{"Op Name": "unmapped", "Task Duration(us)": "0.25"}]
        with tempfile.TemporaryDirectory() as temp:
            prof = Path(temp) / "PROF_000001_20260914160000000_TEST"
            prof.mkdir()
            with patch.object(importer, "find_output_file", side_effect=lambda directory, name: name), \
                 patch.object(importer, "read_csv_rows", side_effect=lambda name: raw_rows if name == "op_summary" else []), \
                 patch.object(importer, "load_perf_data", return_value={"cases": [], "snapshots": [], "runs": [], "models": []}):
                data = importer.import_prof(prof, "gdn", "A2", persist=False, attributes={"batch": 1, "tokens": 128})
        summary = data["snapshots"][0]["op_summary"]
        self.assertEqual(summary[0]["op_name"], "unmapped")
        self.assertEqual(summary[0]["time_stats"]["Task Duration(us)"]["total"], 0.25)
        self.assertEqual(data["runs"][0]["snapshot"]["op_summary"], summary)
        self.assertEqual(data["snapshots"][0]["op_summary_details"]["rows"], [[0, 0.25]])
        self.assertEqual(data["runs"][0]["snapshot"]["op_summary_details"], data["snapshots"][0]["op_summary_details"])

    def test_execution_records_keep_repeated_names_order_precision_and_missing_values(self):
        rows = [
            {"Op Name": "repeat", "Task Duration(us)": "0.125", "Task Start Time(us)": "100"},
            {"Op Name": "middle", "Task Duration(us)": "0", "Task Start Time(us)": "101"},
            {"Op Name": "repeat", "Task Duration(us)": "0.375", "Task Start Time(us)": "102"},
            {"Op Name": "repeat", "Task Duration(us)": "NaN"},
        ]
        result = importer.execution_summary_details(rows)
        self.assertEqual(result["fields"], ["Task Duration(us)", "Task Start Time(us)"])
        self.assertEqual([name["op_name"] for name in result["names"]], ["repeat", "middle"])
        self.assertEqual(result["rows"], [[0, 0.125, 100], [1, 0, 101], [0, 0.375, 102], [0, None, None]])


if __name__ == "__main__":
    unittest.main()
