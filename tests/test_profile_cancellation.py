import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.perf_runner import _run_cancelable_profile
from backend.runner_agent import RunnerAgent, RunnerNotificationClient


class CancelTransportTests(unittest.TestCase):
    @patch("backend.perf_runner._remote_execution_command", side_effect=lambda c, cmd, **kw: cmd)
    @patch("backend.perf_runner._ssh_command", side_effect=lambda c, cmd: ["ssh", cmd])
    @patch("backend.perf_runner.subprocess.run")
    @patch("backend.perf_runner.subprocess.Popen")
    def test_cancellation_retries_until_remote_exit_is_confirmed(self, popen, run, *_):
        event = threading.Event()
        process = popen.return_value
        def communicate(**kw):
            if not event.is_set():
                event.set()
                raise subprocess.TimeoutExpired("ssh", 1)
            return "", ""
        process.communicate.side_effect = communicate
        process.poll.return_value = 0
        run.side_effect = [
            subprocess.CompletedProcess([], 255, "", "connection lost"),
            subprocess.CompletedProcess([], 0, '{"stopped":true}', ""),
        ]
        with self.assertRaisesRegex(InterruptedError, "确认退出"):
            _run_cancelable_profile(None, None, "sleep 60", "/tmp/test", event)
        self.assertEqual(run.call_count, 2)
        process.kill.assert_not_called()

    def test_notification_is_scoped_to_current_attempt(self):
        agent = RunnerAgent.__new__(RunnerAgent)
        event = threading.Event()
        agent._job_cancellations = {("job", "new"): event}
        agent.config = SimpleNamespace(runner_id="runner", notification_enabled=True)
        client = RunnerNotificationClient(agent, connector=Mock())
        def send(attempt, runner="runner"):
            client._handle_message(json.dumps({"version": 1, "type": "job_cancel_requested",
                "runner_id": runner, "job_id": "job", "attempt_id": attempt}))
        send("old")
        send("new", "other")
        self.assertFalse(event.is_set())
        send("new")
        self.assertTrue(event.is_set())


@unittest.skipUnless(sys.platform == "linux", "requires Linux process identity and signals")
class RemoteControlTests(unittest.TestCase):
    helper = Path(__file__).resolve().parents[1] / "backend" / "remote_profile_control.py"

    def command(self, action, directory, token, command=""):
        return [sys.executable, str(self.helper), action, str(directory), token, command]

    def test_cancel_before_start_prevents_any_execution(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "unexpected"
            subprocess.run(self.command("cancel", root, "prestart"), check=True, capture_output=True)
            result = subprocess.run(self.command("run", root, "prestart", f"touch {marker}"), capture_output=True)
            self.assertEqual(result.returncode, 130)
            self.assertFalse(marker.exists())

    def test_cancel_kills_term_resistant_descendant_in_new_session_only(self):
        from backend.remote_profile_control import members
        with tempfile.TemporaryDirectory() as root:
            command = "python3 -c 'import os,signal,subprocess,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([\"sleep\",\"60\"],start_new_session=True); time.sleep(60)'"
            unrelated = subprocess.Popen(["sleep", "60"])
            process = subprocess.Popen(self.command("run", root, "owned-attempt", command),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while len(members("owned-attempt")) < 2 and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertGreaterEqual(len(members("owned-attempt")), 2)
                mismatch = subprocess.run(self.command("cancel", root, "wrong-attempt"), capture_output=True)
                self.assertNotEqual(mismatch.returncode, 0)
                self.assertIsNone(process.poll())
                start = time.monotonic()
                result = subprocess.run(self.command("cancel", root, "owned-attempt"), capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["stopped"])
                self.assertLess(time.monotonic() - start, 8)
                process.communicate(timeout=5)
                self.assertFalse(members("owned-attempt"))
                self.assertIsNone(unrelated.poll())
            finally:
                subprocess.run(self.command("cancel", root, "owned-attempt"), capture_output=True)
                unrelated.kill()
                unrelated.wait()
                process.communicate(timeout=5)

    def test_normal_completion_and_late_cancel_are_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(self.command("run", root, "normal", "true"), capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for _ in range(2):
                result = subprocess.run(self.command("cancel", root, "normal"), capture_output=True)
                self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
