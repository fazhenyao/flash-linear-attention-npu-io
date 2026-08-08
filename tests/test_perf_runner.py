import os
import unittest
from unittest.mock import patch

from backend.perf_runner import (
    _remote_execution_command,
    _remote_output_path,
    _scp_command,
    _ssh_command,
    build_command,
    load_config,
)


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


if __name__ == "__main__":
    unittest.main()
