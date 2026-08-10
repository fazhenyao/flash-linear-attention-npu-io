import os
import subprocess
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.perf_runner import (
    _remote_execution_command,
    _remote_output_path,
    _list_remote_prof_dirs,
    _run_command,
    _scp_command,
    _ssh_command,
    build_command,
    build_profiler_command,
    load_config,
    parse_npu_smi_status,
    prof_output_root,
    resolve_chip,
)
from backend.runner_agent import RunnerAgent
from scripts.cube_theoretical_flops import compute_mfu
from scripts.hbm_theoretical_bytes import compute_mbu


class PerfRunnerRemoteCommandTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "PERF_RUN_MODE": "ssh",
            "PERF_SSH_HOST": "192.168.9.221",
            "PERF_REMOTE_WORKDIR": "/workspace/project",
            "PERF_REMOTE_ENV_SCRIPT": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "PERF_REMOTE_PATH_PREPEND": "/opt/ascend/profiler/bin:/opt/ascend/msopt/bin",
            "PERF_REMOTE_CONDA_SH": "/data/miniconda3/etc/profile.d/conda.sh",
            "PERF_REMOTE_CONDA_ENV": "fla_dump",
        }

    def test_remote_command_loads_environment_before_changing_directory(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        command = _remote_execution_command(config, "msprof --help")

        self.assertEqual(
            command,
            ". /usr/local/Ascend/ascend-toolkit/set_env.sh "
            "&& export PATH=/opt/ascend/profiler/bin:/opt/ascend/msopt/bin:\"$PATH\" "
            "&& . /data/miniconda3/etc/profile.d/conda.sh "
            "&& conda activate fla_dump "
            "&& cd /workspace/project && msprof --help",
        )

    def test_relative_output_is_resolved_from_remote_workdir(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        self.assertEqual(
            _remote_output_path(config, "data/prof_gdr"),
            "/workspace/project/data/prof_gdr",
        )
        self.assertEqual(_remote_output_path(config, "/data/prof_gdr"), "/data/prof_gdr")

    def test_ssh_dry_run_keeps_posix_output_path_on_windows(self):
        with patch.dict(os.environ, self.environment, clear=False):
            command = build_command({"prof_tool": "msprof", "attributes": {}})

        self.assertIn("data/prof_gdr", command)
        self.assertIn("--device 2", command)
        self.assertNotIn("data\\\\prof_gdr", command)

    def test_remote_script_can_be_mapped_by_trusted_runner_config(self):
        remote_script = (
            "/home/npu_user7/fazhenyao/flash-linear-attention-npu/"
            "examples/flash_gated_delta_rule.py"
        )
        environment = {**self.environment, "PERF_REMOTE_SCRIPT": remote_script}

        with patch.dict(os.environ, environment, clear=False):
            command = build_command({"prof_tool": "msprof", "attributes": {}})
            profiler_command = build_profiler_command({"prof_tool": "msprof", "attributes": {}})

        self.assertIn(remote_script, command)
        self.assertNotIn("python3 scripts/flash_gated_delta_rule.py", command)

        self.assertTrue(profiler_command.startswith("msprof --output="))
        self.assertIn(remote_script, profiler_command)
        self.assertNotIn("ssh ", profiler_command)
        self.assertNotIn("conda activate", profiler_command)

    def test_a5_chip_and_instance_local_artifact_roots(self):
        environment = {
            **self.environment,
            "PERF_CHIP": "A5",
            "PERF_LOCAL_PROF_OUTPUT": "data/runner-artifacts/a5/prof_gdr",
            "PERF_LOCAL_OP_OUTPUT": "data/runner-artifacts/a5/prof_op",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = load_config()
            self.assertEqual(resolve_chip({}, config), "A5")
            self.assertEqual(
                prof_output_root("msprof", local=True),
                config.local_script.resolve().parents[1] / "data/runner-artifacts/a5/prof_gdr",
            )

    def test_a5_does_not_reuse_a2_theoretical_limits(self):
        attributes = {
            "batch": 1,
            "query_heads": 1,
            "value_heads": 1,
            "tokens": 64,
            "key_dim": 128,
            "value_dim": 128,
            "chunk_size": 64,
            "dtype": "bf16",
        }

        self.assertIsNone(
            compute_mfu(
                "chunk_fwd_o",
                attributes,
                task_duration_us=10,
                block_dim=1,
                chip="A5",
            )
        )
        self.assertIsNone(
            compute_mbu(
                "chunk_fwd_o",
                attributes,
                task_duration_us=10,
                chip="A5",
            )
        )

    def test_ssh_and_scp_have_connection_timeouts(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        for command in (
            _ssh_command(config, "true"),
            _scp_command(config, "/tmp/prof", os.path.abspath("data")),
        ):
            self.assertIn("ConnectTimeout=10", command)
            self.assertIn("ServerAliveInterval=15", command)
            self.assertIn("ServerAliveCountMax=4", command)

    def test_parse_a2_npu_smi_status_with_process(self):
        output = """__NPU__:0
        NPU ID                         : 0
        Chip Count                     : 1
        HBM Capacity(MB)               : 65536
        HBM Usage Rate(%)              : 5
        Aicore Usage Rate(%)           : 0
        Aivector Usage Rate(%)         : 0
        NPU Utilization(%)             : 0
__PROCESSES__
        NPU ID                         : 0
        Process id:1654497 Process name:python            Process memory(MB):177
        Chip ID                        : 0
__END_NPU__
"""

        device = parse_npu_smi_status(output, [0])[0]

        self.assertTrue(device["available"])
        self.assertEqual(device["status"], "busy")
        self.assertEqual(device["hbm_capacity_mb"], 65536)
        self.assertEqual(device["hbm_used_mb"], 3277)
        self.assertEqual(device["process_count"], 1)
        self.assertEqual(device["processes"][0], {
            "pid": 1654497,
            "name": "python",
            "memory_mb": 177,
        })

    def test_parse_a5_npu_smi_status_and_unavailable_slot(self):
        output = """__NPU__:0
        NPU ID                         : 0
        HBM Capacity(MB)               : 131072
        HBM Usage Rate(%)              : 15
        Aicore Usage Rate(%)           : 0
        Aivector Usage Rate(%)         : 2
        NPU Utilization(%)             : 0
__PROCESSES__
        NPU ID                         : 0
        No process in device.
__END_NPU__
__NPU__:4
        NPU ID                         : 4
Failed to query npu chip: 0 info.
Failed to query "usages" info.
__PROCESSES__
__END_NPU__
"""

        devices = parse_npu_smi_status(output, [0, 4])

        self.assertEqual(devices[0]["status"], "busy")
        self.assertEqual(devices[0]["hbm_capacity_mb"], 131072)
        self.assertEqual(devices[0]["process_count"], 0)
        self.assertFalse(devices[1]["available"])
        self.assertEqual(devices[1]["status"], "unavailable")

    def test_parse_npu_smi_status_returns_all_processes(self):
        process_lines = "\n".join(
            f"Process id:{1000 + index} Process name:python-{index} Process memory(MB):{index + 1}"
            for index in range(20)
        )
        output = f"""__NPU__:0
HBM Capacity(MB) : 65536
HBM Usage Rate(%) : 10
NPU Utilization(%) : 0
__PROCESSES__
{process_lines}
__END_NPU__
"""

        device = parse_npu_smi_status(output, [0])[0]

        self.assertEqual(device["process_count"], 20)
        self.assertEqual(len(device["processes"]), 20)
        self.assertFalse(device["processes_truncated"])
        self.assertEqual(device["processes"][-1]["pid"], 1019)

    @patch("backend.runner_agent.collect_npu_device_status")
    @patch("backend.runner_agent.load_config")
    def test_forced_npu_status_refresh_acknowledges_request(self, load_runner_config, collect_status):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(
            runner_id="relay-test",
            npu_device_count=8,
            npu_status_timeout_seconds=60,
        )
        agent.current_jobs = 0
        agent._npu_status_lock = threading.Lock()
        agent._npu_status_refreshing = True
        agent._npu_status_pending_refresh_id = None
        agent._npu_status_checked_at = 0.0
        agent._npu_status = {"updated_at": None, "devices": [], "error": ""}
        agent.health = Mock(return_value={"vpn_connected": True, "npu_reachable": True})
        agent.send_runner_heartbeat = Mock()
        load_runner_config.return_value = Mock()
        collect_status.return_value = [{"id": 0, "available": True, "processes": []}]

        agent._refresh_npu_status("refresh-test")

        self.assertEqual(agent._npu_status["refresh_request_id"], "refresh-test")
        self.assertFalse(agent._npu_status_refreshing)
        agent.send_runner_heartbeat.assert_called_once()

    @patch("backend.perf_runner.subprocess.run")
    def test_background_commands_do_not_inherit_stdin(self, run):
        run.return_value = Mock()

        _run_command(["ssh", "example", "true"])

        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    @patch("backend.perf_runner._run_command")
    def test_remote_scan_preserves_case_sensitive_prof_prefix(self, run_command):
        run_command.return_value = Mock(
            stdout="/workspace/project/data/prof_gdr/PROF_000001\n"
        )
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        self.assertEqual(_list_remote_prof_dirs(config, "msprof"), {"PROF_000001"})
        remote_command = run_command.call_args.args[0][-1]
        self.assertIn("ls -1d", remote_command)
        self.assertIn("/PROF_*", remote_command)

    @patch("backend.runner_agent.JobHeartbeat")
    @patch("backend.runner_agent.execute")
    def test_runner_agent_does_not_persist_repository_data(self, execute, heartbeat_class):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.api = Mock()
        agent.save_job_state = Mock()
        agent.build_artifacts = Mock(return_value=([], []))
        agent.environment_summary = Mock(return_value={})
        agent.send_runner_heartbeat = Mock()
        agent.health = Mock(return_value={})
        heartbeat_class.return_value.cancel_requested.is_set.return_value = False
        execute.return_value = {
            "snapshot": {},
            "data": {},
            "profiler_command": "msprof --output=data/prof_gdr python3 test.py",
        }
        job = {
            "id": "job-test",
            "attempt_id": "attempt-test",
            "lease_token": "lease-test",
            "request": {"prof_tool": "msprof"},
        }

        agent.run_job(job)

        execute.assert_called_once_with(
            {"prof_tool": "msprof"},
            persist_local_data=False,
        )
        complete_payload = next(
            call.args[1]
            for call in agent.api.post.call_args_list
            if call.args[0].endswith("/complete")
        )
        self.assertEqual(
            complete_payload["profiler_command"],
            "msprof --output=data/prof_gdr python3 test.py",
        )

    def test_runner_agent_archives_prof_directory_with_root_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "PROF_000001"
            (source / "mindstudio_profiler_output").mkdir(parents=True)
            (source / "mindstudio_profiler_output" / "summary.csv").write_text(
                "name,duration\noperator,1\n",
                encoding="utf-8",
            )
            agent = RunnerAgent.__new__(RunnerAgent)
            agent.config = SimpleNamespace(state_dir=root / "state")

            archive_path = agent.create_artifact_archive("perf-job-test", source)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["PROF_000001/mindstudio_profiler_output/summary.csv"],
                )

    def test_runner_agent_uploads_archive_in_configured_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "artifact.zip"
            archive_path.write_bytes(b"abcdefghijkl")
            agent = RunnerAgent.__new__(RunnerAgent)
            agent.config = SimpleNamespace(runner_id="relay-test", upload_part_bytes=5)
            agent.api = Mock()

            def post(path, _payload):
                if path.endswith("/start"):
                    return {"upload_id": "upload-test", "artifact_id": "artifact-test"}
                if path.endswith("/complete"):
                    return {"artifact": {"id": "artifact-test", "storage": "r2"}}
                return {"ok": True}

            agent.api.post.side_effect = post
            agent.api.put.side_effect = [
                {"part": {"partNumber": 1, "etag": "etag-1"}},
                {"part": {"partNumber": 2, "etag": "etag-2"}},
                {"part": {"partNumber": 3, "etag": "etag-3"}},
            ]
            job = {
                "id": "perf-job-test",
                "attempt_id": "attempt-test",
                "lease_token": "lease-test",
            }

            artifact = agent.multipart_upload(
                job,
                archive_path,
                {"id": "artifact-test", "filename": "artifact.zip"},
            )

            self.assertEqual(artifact["storage"], "r2")
            self.assertEqual(agent.api.put.call_count, 3)


if __name__ == "__main__":
    unittest.main()
