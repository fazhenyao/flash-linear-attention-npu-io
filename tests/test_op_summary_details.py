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


if __name__ == "__main__":
    unittest.main()
