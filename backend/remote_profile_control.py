"""Linux task supervisor, also executed on the NPU host via python -c.

Each attempt has a unique token inherited by its processes. Cancellation keeps
a tombstone under a file lock so it also works before the supervisor starts.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

TAG = "FLA_PROFILE_ATTEMPT"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # Linux starttime, unaffected by PID reuse.
    except (OSError, IndexError):
        return None


def members(token):
    expected = f"{TAG}={token}".encode()
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            start = identity(entry.name)
            if start and expected in (entry / "environ").read_bytes().split(b"\0"):
                found[int(entry.name)] = start
        except (OSError, ProcessLookupError):
            continue
    return found


def terminate(token, grace=3.0, timeout=10.0):
    deadline = time.monotonic() + timeout
    escalate_at = time.monotonic() + grace
    sent_term = set()
    while True:
        current = members(token)
        if not current:
            return True
        for pid, start in current.items():
            sig = signal.SIGKILL if time.monotonic() >= escalate_at else signal.SIGTERM
            if sig == signal.SIGTERM and (pid, start) in sent_term:
                continue
            # Revalidate identity before every signal. Include descendants that
            # created new sessions: killing only the SSH client is insufficient.
            if identity(pid) != start:
                continue
            try:
                os.kill(pid, sig)
                sent_term.add((pid, start))
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def control(action, directory, token, command=""):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = root / "token"
        if owner.exists() and owner.read_text() != token:
            raise RuntimeError("attempt identity mismatch")
        owner.write_text(token)
        if action == "cancel":
            (root / "cancel").touch()
        else:
            if (root / "cancel").exists():
                return 130
            if (root / "started").exists():
                raise RuntimeError("attempt already started")
            child = subprocess.Popen(
                ["/bin/bash", "-c", "exec " + command],
                start_new_session=True, env={**os.environ, TAG: token},
            )
            (root / "started").write_text(json.dumps({"pid": child.pid, "start": identity(child.pid)}))
        fcntl.flock(lock, fcntl.LOCK_UN)
    if action == "cancel":
        stopped = terminate(token)
        print(json.dumps({"stopped": stopped}), flush=True)
        return 0 if stopped else 3
    code = child.wait()
    # A profiler may exit before its children. Do not acknowledge completion or
    # cancellation while an attempt-owned process can still execute on the NPU.
    if not terminate(token):
        return 3
    (root / "exit_code").write_text(str(code))
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "cancel"])
    parser.add_argument("directory")
    parser.add_argument("token")
    parser.add_argument("command", nargs="?", default="")
    args = parser.parse_args()
    raise SystemExit(control(args.action, args.directory, args.token, args.command))
