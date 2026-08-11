import os
import base64
import json
import subprocess
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.perf_runner import (
    _activate_remote_source_build,
    _parse_persistent_build_status,
    _prepare_remote_source_build,
    _resolve_remote_deployed_source,
    _remote_execution_command,
    _remote_output_path,
    _list_remote_prof_dirs,
    _run_command,
    _scp_command,
    _ssh_command,
    build_command,
    build_profiler_command,
    execute,
    list_remote_source_branches,
    load_config,
    parse_npu_smi_status,
    persistent_build_handle,
    poll_persistent_build_install,
    prof_output_root,
    RemoteBuildOrphanedError,
    RemoteConnectionError,
    resolve_chip,
    resolve_execution_environment,
    soc_build_target,
    start_persistent_build_install,
)
from backend.runner_agent import AgentConfig, RunnerAgent
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
            "PERF_REMOTE_SOURCE_REPO": "/workspace/user/flash-linear-attention-npu",
            "PERF_ALLOWED_CANN_ROOTS": "/usr/local/Ascend;/data/user/cann",
            "PERF_ALLOWED_SOURCE_ROOTS": "/workspace/user",
            "PERF_REMOTE_BUILD_ROOT": "/tmp/fla-runner-builds",
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

    def test_custom_execution_environment_is_limited_to_runner_roots(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()
        payload = {
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "feature/a5",
            }
        }

        execution = resolve_execution_environment(payload, config)

        self.assertEqual(execution.env_script, "/data/user/cann/7_20/ascend-toolkit/set_env.sh")
        self.assertEqual(execution.conda_env, "feature_env")
        with self.assertRaisesRegex(ValueError, "源码仓库路径不在"):
            resolve_execution_environment({
                "execution_environment": {
                    **payload["execution_environment"],
                    "source_repo": "/etc/flash-linear-attention-npu",
                }
            }, config)

    def test_npu_status_automatic_refresh_defaults_to_thirty_minutes(self):
        with patch.dict(
            os.environ,
            {"RUNNER_TOKEN": "test-token", "RUNNER_ID": "relay-test"},
            clear=True,
        ):
            config = AgentConfig.from_env()

        self.assertEqual(config.npu_status_interval_seconds, 30 * 60)

    @patch("backend.perf_runner._run_remote_checked")
    def test_source_branch_query_returns_local_and_remote_branches(self, run_remote):
        run_remote.return_value = Mock(
            stdout=(
                "refs/heads/main\n"
                "refs/heads/feature/a5\n"
                "refs/remotes/origin/HEAD\n"
                "refs/remotes/origin/main\n"
                "refs/remotes/origin/feature/remote\n"
                "refs/remotes/origin/feature/../invalid\n"
            )
        )
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        branches = list_remote_source_branches(
            config,
            "/workspace/user/flash-linear-attention-npu",
        )

        self.assertEqual(branches, [
            {"source": "local", "name": "feature/a5"},
            {"source": "local", "name": "main"},
            {"source": "remote", "name": "feature/remote"},
            {"source": "remote", "name": "main"},
        ])
        list_command = run_remote.call_args_list[0].args[1]
        refresh_command = run_remote.call_args_list[1].args[1]
        self.assertNotIn("fetch --prune origin", list_command)
        self.assertIn("refs/heads", list_command)
        self.assertIn("refs/remotes/origin", list_command)
        self.assertIn("timeout --signal=TERM --kill-after=2s 8s", refresh_command)
        self.assertIn("fetch --prune origin", refresh_command)
        with self.assertRaisesRegex(ValueError, "不在 Relay 允许目录内"):
            list_remote_source_branches(config, "/etc/private-repo")

    @patch("backend.perf_runner._run_remote_checked")
    def test_source_branch_query_keeps_existing_refs_when_origin_refresh_fails(self, run_remote):
        run_remote.side_effect = [
            SimpleNamespace(stdout="refs/heads/main\nrefs/remotes/origin/cached\n"),
            RuntimeError("origin unavailable"),
        ]
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()

        branches = list_remote_source_branches(config, "/workspace/user/flash-linear-attention-npu")

        self.assertEqual(branches, [
            {"source": "local", "name": "main"},
            {"source": "remote", "name": "cached"},
        ])

    def test_custom_remote_command_uses_selected_cann_and_conda(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()
        execution = resolve_execution_environment({
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": False,
                "branch": "",
            }
        }, config)

        command = _remote_execution_command(config, "python --version", execution=execution)

        self.assertIn(". /data/user/cann/7_20/ascend-toolkit/set_env.sh", command)
        self.assertIn("conda activate feature_env", command)

    def test_profile_can_select_a_previously_deployed_branch(self):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()
        execution = resolve_execution_environment({
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": False,
                "branch": "feature/a5",
            }
        }, config)

        self.assertFalse(execution.rebuild)
        self.assertEqual(execution.branch, "feature/a5")
        self.assertEqual(execution.branch_source, "local")

    @patch("backend.perf_runner._run_remote_checked")
    def test_source_build_uses_readme_commands_and_chip_soc(self, run_remote):
        run_remote.side_effect = [
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout="Successfully installed\n"),
        ]
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        execution = resolve_execution_environment({
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "feature/a5",
            }
        }, config)

        result = _prepare_remote_source_build(config, execution, config.chip)

        prepare_command = run_remote.call_args_list[0].args[1]
        build_command = run_remote.call_args_list[1].args[1]
        self.assertIn("refs/heads/feature/a5^{commit}", prepare_command)
        self.assertNotIn("fetch --prune origin", prepare_command)
        self.assertIn("python scripts/check_npu_env.py --build-only", build_command)
        self.assertIn("FLA_NPU_SOC=ascend950", build_command)
        self.assertIn("python -m pip wheel --no-build-isolation --no-deps", build_command)
        self.assertIn("--force-reinstall --no-cache-dir --no-deps", build_command)
        self.assertEqual(result["commit"], "a" * 40)

    @patch("backend.perf_runner._run_remote_checked")
    def test_remote_source_build_refreshes_origin_and_uses_cached_ref_offline(self, run_remote):
        run_remote.side_effect = [
            SimpleNamespace(stdout="b" * 40 + "\n"),
            SimpleNamespace(stdout="Successfully installed\n"),
        ]
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        execution = resolve_execution_environment({
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "feature/remote",
                "branch_source": "remote",
            }
        }, config)

        result = _prepare_remote_source_build(config, execution, config.chip)

        prepare_command = run_remote.call_args_list[0].args[1]
        self.assertIn("refs/remotes/origin/feature/remote^{commit}", prepare_command)
        self.assertIn("fetch --prune origin", prepare_command)
        self.assertIn("|| true", prepare_command)
        self.assertEqual(result["branch_source"], "remote")

    def test_build_soc_is_fixed_by_runner_chip(self):
        self.assertEqual(soc_build_target("A2"), "ascend910b")
        self.assertEqual(soc_build_target("A3"), "ascend910_93")
        self.assertEqual(soc_build_target("A5"), "ascend950")

    def test_persistent_build_handle_isolated_by_attempt(self):
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        payload = {
            "task_type": "build_install",
            "chip": "A5",
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "feature/a5",
            },
        }

        handle = persistent_build_handle(payload, "attempt-12345678", config)

        self.assertEqual(handle["execution_id"], "attempt-12345678")
        self.assertEqual(handle["worktree"], "/tmp/fla-runner-builds/worktrees/attempt-12345678")
        self.assertEqual(handle["control_dir"], "/tmp/fla-runner-builds/controls/attempt-12345678")
        self.assertNotIn(handle["control_dir"], handle["worktree"])
        self.assertEqual(handle["log_path"], f"{handle['control_dir']}/build.log")

    def test_persistent_build_status_decodes_log_and_exit_code(self):
        log = "building operator\ncompiler failed\n"
        output = "\n".join([
            "__STATE__=failed",
            "__PID__=123",
            "__ALIVE__=0",
            "__EXIT_CODE__=2",
            f"__LOG_SIZE__={len(log)}",
            f"__LOG_BASE64__={base64.b64encode(log.encode()).decode()}",
        ])

        status = _parse_persistent_build_status(output, {"worktree": "/tmp/build"})

        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["exit_code"], 2)
        self.assertEqual(status["log_tail"], log)

    def test_persistent_build_status_tracks_live_process_when_state_is_missing(self):
        output = "\n".join([
            "__STATE__=missing",
            "__PID__=123",
            "__ALIVE__=1",
            "__EXIT_CODE__=",
            "__LOG_SIZE__=0",
            "__LOG_BASE64__=",
        ])

        status = _parse_persistent_build_status(output, {"worktree": "/tmp/build"})

        self.assertEqual(status["state"], "running")
        self.assertTrue(status["active"])
        self.assertIn("without a state file", status["log_tail"])

    @patch("backend.perf_runner._run_remote_checked", side_effect=RemoteConnectionError("VPN unavailable"))
    def test_persistent_build_poll_does_not_treat_ssh_failure_as_missing(self, _run_remote):
        with patch.dict(os.environ, self.environment, clear=False):
            config = load_config()
        handle = {
            "control_dir": "/tmp/fla-runner-builds/attempts/attempt-12345678/.fla-runner",
        }

        with self.assertRaisesRegex(RemoteConnectionError, "VPN unavailable"):
            poll_persistent_build_install(handle, config, allow_missing=True)

    @patch("backend.perf_runner._run_remote_checked")
    @patch("backend.perf_runner._persistent_build_script", return_value="#!/bin/bash\nexit 0\n")
    @patch("backend.perf_runner._cleanup_remote_source_build")
    @patch("backend.perf_runner._validate_remote_execution_environment")
    @patch("backend.perf_runner.poll_persistent_build_install")
    def test_persistent_build_starts_only_the_script_in_background(
        self,
        poll,
        _validate,
        _cleanup,
        _script,
        run_remote,
    ):
        poll.return_value = {"state": "missing"}
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        payload = {
            "task_type": "build_install",
            "chip": "A5",
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "main",
            },
        }
        handle = persistent_build_handle(payload, "attempt-12345678", config)

        start_persistent_build_install(payload, handle, config)

        command = run_remote.call_args.args[1]
        self.assertIn("&& { setsid nohup", command)
        self.assertIn("starter=$!", command)
        self.assertTrue(command.endswith("; }"))
        _cleanup.assert_called_once()

    def test_persistent_build_script_builds_once_and_uses_environment_lock(self):
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        payload = {
            "task_type": "build_install",
            "chip": "A5",
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "main",
            },
        }
        handle = persistent_build_handle(payload, "attempt-12345678", config)
        execution = resolve_execution_environment(payload, config)

        from backend.perf_runner import _persistent_build_script
        script = _persistent_build_script(config, execution, "A5", handle)

        self.assertEqual(script.count("python -m pip wheel"), 1)
        self.assertEqual(script.count("python -m pip install"), 1)
        self.assertIn('exec 9>"$lock_path"', script)
        self.assertIn("flock 9", script)
        self.assertIn("/controls/attempt-12345678", script)
        self.assertIn("/worktrees/attempt-12345678", script)

    def test_persistent_build_status_recovers_success_from_active_marker(self):
        output = "\n".join([
            "__STATE__=succeeded",
            "__PID__=",
            "__ALIVE__=0",
            "__EXIT_CODE__=0",
            "__LOG_SIZE__=0",
            "__LOG_BASE64__=",
            f"__COMMIT__={'d' * 40}",
            "__DEPLOYMENT__=/tmp/fla-runner-builds/active/environment",
            "__RECOVERED__=1",
        ])

        status = _parse_persistent_build_status(output, {"execution_id": "attempt-12345678"})

        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(status["commit"], "d" * 40)
        self.assertTrue(status["recovered_from_deployment"])

    def test_missing_persistent_build_result_is_orphaned(self):
        with self.assertRaisesRegex(RemoteBuildOrphanedError, "无法确认"):
            from backend.perf_runner import persistent_build_result
            persistent_build_result({}, {"state": "missing"})

    @patch("backend.perf_runner._run_remote_checked")
    def test_build_acknowledgement_keeps_three_deployments(self, run_remote):
        from backend.perf_runner import acknowledge_persistent_build_install

        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        acknowledge_persistent_build_install({
            "control_dir": "/tmp/fla-runner-builds/controls/attempt-12345678",
            "deployment": "/tmp/fla-runner-builds/active/environment",
            "source_repo": "/workspace/user/flash-linear-attention-npu",
        }, config)

        command = run_remote.call_args.args[1]
        self.assertIn("acknowledged.tmp", command)
        self.assertIn('[ "$kept" -le 3 ]', command)
        self.assertIn('"$worktrees_root"/*', command)

    @patch("backend.perf_runner._run_remote_checked")
    def test_activated_build_is_resolved_for_later_profile(self, run_remote):
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        execution = resolve_execution_environment({
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": False,
                "branch": "feature/a5",
            }
        }, config)
        build_info = {
            "worktree": "/tmp/fla-runner-builds/build-1",
            "commit": "a" * 40,
            "soc": "ascend950",
        }
        run_remote.side_effect = [
            SimpleNamespace(stdout="\n"),
            SimpleNamespace(stdout=f"feature/a5\nlocal\n{'a' * 40}\nascend950\n"),
        ]

        deployment = _activate_remote_source_build(config, execution, "A5", build_info)
        resolved = _resolve_remote_deployed_source(config, execution, "A5")

        self.assertEqual(resolved, deployment)
        self.assertIn("ln -sfn", run_remote.call_args_list[0].args[1])
        self.assertIn(".fla-runner-deployment", run_remote.call_args_list[1].args[1])

    @patch("backend.perf_runner._activate_remote_source_build")
    @patch("backend.perf_runner._prepare_remote_source_build")
    @patch("backend.perf_runner._validate_remote_execution_environment")
    @patch("backend.perf_runner.ensure_runner_configured")
    def test_build_install_task_does_not_execute_profiler(self, configured, validate, prepare, activate):
        with patch.dict(os.environ, {**self.environment, "PERF_CHIP": "A5"}, clear=False):
            config = load_config()
        configured.return_value = config
        prepare.return_value = {
            "worktree": "/tmp/fla-runner-builds/build-1",
            "commit": "b" * 40,
            "soc": "ascend950",
        }
        activate.return_value = "/tmp/fla-runner-builds/active/current"

        result = execute({
            "task_type": "build_install",
            "chip": "A5",
            "execution_environment": {
                "cann_path": "/data/user/cann/7_20/ascend-toolkit",
                "conda_env": "feature_env",
                "source_repo": "/workspace/user/flash-linear-attention-npu",
                "rebuild": True,
                "branch": "feature/a5",
            },
        }, persist_local_data=False)

        self.assertEqual(result["task_type"], "build_install")
        self.assertNotIn("profiler_command", result)
        validate.assert_called_once()
        prepare.assert_called_once()
        activate.assert_called_once()

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

    @patch("backend.runner_agent.list_remote_source_branches")
    @patch("backend.runner_agent.load_config")
    def test_source_branch_refresh_acknowledges_request(self, load_runner_config, list_branches):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.current_jobs = 0
        agent._source_branches_lock = threading.Lock()
        agent._source_branches_refreshing = True
        agent._source_branches_pending = None
        agent._source_branches = {"branches": [], "error": ""}
        agent.health = Mock(return_value={"vpn_connected": True, "npu_reachable": True})
        agent.send_runner_heartbeat = Mock()
        load_runner_config.return_value = Mock()
        list_branches.return_value = [
            {"source": "local", "name": "main"},
            {"source": "remote", "name": "feature/a5"},
        ]

        agent._refresh_source_branches("branches-test", "/workspace/user/repo")

        self.assertEqual(agent._source_branches["refresh_request_id"], "branches-test")
        self.assertEqual(agent._source_branches["branches"], [
            {"source": "local", "name": "main"},
            {"source": "remote", "name": "feature/a5"},
        ])
        self.assertFalse(agent._source_branches_refreshing)
        agent.send_runner_heartbeat.assert_called_once()

    @patch("backend.runner_agent.list_remote_source_branches")
    @patch("backend.runner_agent.load_config")
    def test_source_branch_refresh_failure_preserves_cached_branches(self, load_runner_config, list_branches):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.current_jobs = 0
        agent._source_branches_lock = threading.Lock()
        agent._source_branches_refreshing = True
        agent._source_branches_pending = None
        cached = [{"source": "local", "name": "main"}]
        agent._source_branches = {
            "source_repo": "/workspace/user/repo",
            "branches": cached,
            "error": "",
        }
        agent.health = Mock(return_value={"vpn_connected": False, "npu_reachable": False})
        agent.send_runner_heartbeat = Mock()
        load_runner_config.return_value = Mock()
        list_branches.side_effect = RuntimeError("SSH unavailable")

        agent._refresh_source_branches("branches-failed", "/workspace/user/repo")

        self.assertEqual(agent._source_branches["branches"], cached)
        self.assertEqual(agent._source_branches["error"], "SSH unavailable")
        self.assertTrue(agent._source_branches["stale"])
        self.assertFalse(agent._source_branches_refreshing)
        agent.send_runner_heartbeat.assert_called_once()

    def test_source_branch_cache_survives_agent_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(runner_id="relay-test", state_dir=Path(temp_dir))
            agent = RunnerAgent(config)
            cached = [{"source": "remote", "name": "feature/cached"}]
            agent._save_source_branches_cache({
                "checked_at": "2026-08-11T00:00:00Z",
                "source_repo": "/workspace/user/repo",
                "branches": cached,
            })

            restored = RunnerAgent(config)

            self.assertEqual(restored._source_branches["branches"], cached)
            self.assertEqual(restored._source_branches["source_repo"], "/workspace/user/repo")
            self.assertTrue(restored._source_branches["stale"])

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

    @patch("backend.runner_agent.JobHeartbeat")
    def test_runner_agent_completes_build_without_prof_artifacts(self, heartbeat_class):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.api = Mock()
        agent.save_job_state = Mock()
        agent.build_artifacts = Mock(return_value=([], []))
        agent.environment_summary = Mock(return_value={})
        agent.send_runner_heartbeat = Mock()
        agent.health = Mock(return_value={})
        agent.run_persistent_build_job = Mock(return_value={
            "task_type": "build_install",
            "message": "源码分支 main 编译安装完成",
            "execution_environment": {"branch": "main", "commit": "c" * 40},
        })
        heartbeat_class.return_value.cancel_requested.is_set.return_value = False
        job = {
            "id": "job-build",
            "attempt_id": "attempt-build",
            "lease_token": "lease-build",
            "request": {"task_type": "build_install"},
        }

        agent.run_job(job)

        agent.run_persistent_build_job.assert_called_once()
        complete_payload = next(
            call.args[1]
            for call in agent.api.post.call_args_list
            if call.args[0].endswith("/complete")
        )
        self.assertEqual(complete_payload["task_type"], "build_install")
        self.assertEqual(complete_payload["artifacts"], [])
        self.assertEqual(complete_payload["snapshot"], {})

    def test_runner_agent_recovers_persisted_build_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            jobs_dir = state_dir / "jobs"
            jobs_dir.mkdir()
            record = {
                "job_id": "perf-job-recover",
                "attempt_id": "attempt-recover",
                "lease_token": "lease-recover",
                "state": "running",
                "request": {"task_type": "build_install", "chip": "A5"},
                "remote_build": {"worktree": "/tmp/build-recover"},
            }
            (jobs_dir / "perf-job-recover.json").write_text(json.dumps(record), encoding="utf-8")
            agent = RunnerAgent.__new__(RunnerAgent)
            agent.config = SimpleNamespace(state_dir=state_dir)
            agent.run_job = Mock()

            recovered = agent.recover_build_jobs()

            self.assertEqual(recovered, 1)
            recovered_job = agent.run_job.call_args.args[0]
            self.assertEqual(recovered_job["lease_token"], "lease-recover")
            self.assertEqual(recovered_job["remote_build"]["worktree"], "/tmp/build-recover")
            self.assertTrue(agent.run_job.call_args.kwargs["resume"])

    @patch("backend.runner_agent.JobHeartbeat")
    def test_build_completion_api_failure_stays_reporting_without_marking_failed(self, heartbeat_class):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.api = Mock()
        agent.save_job_state = Mock()
        agent.build_artifacts = Mock(return_value=([], []))
        agent.environment_summary = Mock(return_value={})
        agent.send_runner_heartbeat = Mock()
        agent.health = Mock(return_value={})
        agent.run_persistent_build_job = Mock(return_value={
            "task_type": "build_install",
            "message": "编译安装完成",
            "execution_environment": {"branch": "main", "commit": "c" * 40},
        })
        heartbeat_class.return_value.cancel_requested.is_set.return_value = False

        def post(path, _payload):
            if path.endswith("/complete"):
                raise RuntimeError("Worker temporarily unavailable")
            return {}

        agent.api.post.side_effect = post
        job = {
            "id": "job-report-retry",
            "attempt_id": "attempt-report-retry",
            "lease_token": "lease-report-retry",
            "request": {"task_type": "build_install"},
        }

        agent.run_job(job)

        states = [call.args[1] for call in agent.save_job_state.call_args_list]
        self.assertIn("reporting", states)
        self.assertNotIn("failed", states)
        self.assertNotIn("completed", states)
        posted_paths = [call.args[0] for call in agent.api.post.call_args_list]
        self.assertFalse(any(path.endswith("/fail") for path in posted_paths))

    @patch("backend.runner_agent.JobHeartbeat")
    def test_missing_remote_build_is_reconciled_as_orphaned(self, heartbeat_class):
        agent = RunnerAgent.__new__(RunnerAgent)
        agent.config = SimpleNamespace(runner_id="relay-test")
        agent.api = Mock()
        agent.save_job_state = Mock()
        agent.send_runner_heartbeat = Mock()
        agent.health = Mock(return_value={})
        agent.run_persistent_build_job = Mock(side_effect=RemoteBuildOrphanedError("远端状态无法确认"))
        heartbeat_class.return_value.cancel_requested.is_set.return_value = False
        job = {
            "id": "job-orphaned",
            "attempt_id": "attempt-orphaned",
            "lease_token": "lease-orphaned",
            "request": {"task_type": "build_install"},
            "remote_build": {"control_dir": "/tmp/fla-runner-builds/controls/attempt-orphaned"},
        }

        agent.run_job(job)

        posted_paths = [call.args[0] for call in agent.api.post.call_args_list]
        self.assertTrue(any(path.endswith("/reconcile") for path in posted_paths))
        self.assertFalse(any(path.endswith("/fail") for path in posted_paths))
        states = [call.args[1] for call in agent.save_job_state.call_args_list]
        self.assertIn("reporting_orphaned", states)
        self.assertIn("orphaned", states)

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
