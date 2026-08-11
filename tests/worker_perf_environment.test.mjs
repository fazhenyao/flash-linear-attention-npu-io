import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizePerfExecutionEnvironment,
  normalizePerfJobRequest,
  normalizeSourceBranchesRefreshRequest,
  perfJobCancelTransition,
  runnerCanExecute,
} from "../cloudflare/worker.js";

test("normalizes a controlled source build environment", () => {
  assert.deepEqual(normalizePerfExecutionEnvironment({
    cann_path: "/home/user/cann/ascend-toolkit/",
    conda_env: "f30077529",
    source_repo: "/home/user/flash-linear-attention-npu/",
    rebuild: true,
    branch: "feature/a5-build",
    branch_source: "remote",
  }), {
    cann_path: "/home/user/cann/ascend-toolkit",
    conda_env: "f30077529",
    source_repo: "/home/user/flash-linear-attention-npu",
    rebuild: true,
    branch: "feature/a5-build",
    branch_source: "remote",
  });
});

test("rejects path traversal and unsupported fields", () => {
  assert.throws(() => normalizePerfExecutionEnvironment({
    cann_path: "/home/user/cann/../private",
    conda_env: "fla",
    source_repo: "/home/user/repo",
    rebuild: false,
  }), /invalid cann_path/);
  assert.throws(() => normalizePerfExecutionEnvironment({
    cann_path: "/home/user/cann",
    conda_env: "fla",
    source_repo: "/home/user/repo",
    rebuild: false,
    command: "whoami",
  }), /unsupported execution environment field/);
});

test("requires an explicit valid branch for source rebuilds", () => {
  const base = {
    cann_path: "/home/user/cann",
    conda_env: "fla",
    source_repo: "/home/user/repo",
  };
  assert.throws(() => normalizePerfExecutionEnvironment({ ...base, rebuild: true }), /branch is required/);
  assert.throws(
    () => normalizePerfExecutionEnvironment({ ...base, rebuild: true, branch: "feature/../main" }),
    /invalid source branch/,
  );
  assert.throws(
    () => normalizePerfExecutionEnvironment({ ...base, rebuild: "false" }),
    /rebuild must be a boolean/,
  );
  assert.throws(
    () => normalizePerfExecutionEnvironment({ ...base, rebuild: true, branch: "main", branch_source: "upstream" }),
    /invalid branch_source/,
  );
});

test("allows profile tasks to select an already deployed branch", () => {
  const environment = normalizePerfExecutionEnvironment({
    cann_path: "/home/user/cann",
    conda_env: "fla",
    source_repo: "/home/user/repo",
    rebuild: false,
    branch: "feature/a5",
    branch_source: "remote",
  }, { allowBranchWithoutRebuild: true });
  assert.equal(environment.branch, "feature/a5");
  assert.equal(environment.branch_source, "remote");
  assert.equal(environment.rebuild, false);
});

test("normalizes admin build-install tasks and rejects ordinary users", () => {
  const payload = {
    task_type: "build_install",
    chip: "A5",
    device: 7,
    target_runner_id: "runner-a5",
    execution_environment: {
      cann_path: "/home/user/cann",
      conda_env: "fla",
      source_repo: "/home/user/repo",
      rebuild: true,
      branch: "main",
      branch_source: "remote",
    },
  };
  const request = normalizePerfJobRequest(payload, { role: "admin" });
  assert.equal(request.task_type, "build_install");
  assert.equal(request.prof_tool, "build_install");
  assert.equal(request.script_id, "source-build");
  assert.equal(request.execution_environment.branch_source, "remote");
  assert.throws(() => normalizePerfJobRequest(payload, { role: "user" }), /requires admin role/);
});

test("routes custom environments only to capable runners", () => {
  const request = {
    target_runner_id: "runner-a5",
    chip: "A5",
    device: 7,
    prof_tool: "msprof",
    execution_environment: { rebuild: true },
  };
  const base = {
    chip: "A5",
    devices: [7],
    prof_tools: ["msprof"],
  };
  assert.equal(runnerCanExecute("runner-a5", base, request), false);
  assert.equal(runnerCanExecute("runner-a5", {
    ...base,
    execution_environment: { customizable: true, source_build: false },
  }, request), false);
  assert.equal(runnerCanExecute("runner-a5", {
    ...base,
    execution_environment: { customizable: true, source_build: true },
  }, request), true);
});

test("routes build-install tasks only to explicitly compatible runners", () => {
  const request = {
    task_type: "build_install",
    target_runner_id: "runner-a5",
    chip: "A5",
    execution_environment: { rebuild: true, branch: "main" },
  };
  const base = {
    chip: "A5",
    devices: [7],
    execution_environment: { customizable: true, source_build: true },
  };
  assert.equal(runnerCanExecute("runner-a5", base, request), false);
  assert.equal(runnerCanExecute("runner-a5", {
    ...base,
    job_types: ["profile", "build_install"],
  }, request), true);
});

test("routes remote branch tasks only to remote-branch capable runners", () => {
  const request = {
    task_type: "build_install",
    target_runner_id: "runner-a5",
    chip: "A5",
    execution_environment: { rebuild: true, branch: "main", branch_source: "remote" },
  };
  const base = {
    chip: "A5",
    job_types: ["profile", "build_install"],
    execution_environment: { customizable: true, source_build: true },
  };
  assert.equal(runnerCanExecute("runner-a5", base, request), false);
  assert.equal(runnerCanExecute("runner-a5", {
    ...base,
    execution_environment: { ...base.execution_environment, source_remote_branch_query: true },
  }, request), true);
});

test("requires deployment capability when profiling a selected branch", () => {
  const request = {
    task_type: "profile",
    target_runner_id: "runner-a5",
    chip: "A5",
    device: 7,
    prof_tool: "msprof",
    execution_environment: { rebuild: false, branch: "main" },
  };
  const base = {
    chip: "A5",
    devices: [7],
    prof_tools: ["msprof"],
    job_types: ["profile"],
    execution_environment: { customizable: true, source_build: true },
  };
  assert.equal(runnerCanExecute("runner-a5", base, request), false);
  assert.equal(runnerCanExecute("runner-a5", {
    ...base,
    execution_environment: { ...base.execution_environment, source_deployment: true },
  }, request), true);
});

test("normalizes controlled source branch refresh requests", () => {
  assert.deepEqual(normalizeSourceBranchesRefreshRequest({
    runner_id: "runner-a5",
    source_repo: "/home/user/flash-linear-attention-npu/",
  }), {
    runner_id: "runner-a5",
    source_repo: "/home/user/flash-linear-attention-npu",
  });
  assert.throws(() => normalizeSourceBranchesRefreshRequest({
    runner_id: "runner-a5",
    source_repo: "/home/user/../private",
  }), /invalid source_repo/);
});

test("cancels queued performance jobs immediately", () => {
  assert.deepEqual(perfJobCancelTransition({ status: "queued" }), {
    status: "canceled",
    message: "任务已取消",
    event_type: "canceled",
    level: "info",
  });
});

test("requests cancellation while a runner lease is active", () => {
  const transition = perfJobCancelTransition({
    status: "running",
    lease_expires_at: "2026-08-11T12:01:00.000Z",
  }, Date.parse("2026-08-11T12:00:00.000Z"));
  assert.equal(transition.status, "cancel_requested");
  assert.equal(transition.event_type, "cancel_requested");
});

test("marks unconfirmed cancellation as orphaned after its lease expires", () => {
  for (const leaseExpiresAt of [null, "2026-08-11T11:59:00.000Z"]) {
    const transition = perfJobCancelTransition({
      status: "cancel_requested",
      lease_expires_at: leaseExpiresAt,
    }, Date.parse("2026-08-11T12:00:00.000Z"));
    assert.equal(transition.status, "orphaned");
    assert.equal(transition.event_type, "cancel_unconfirmed");
    assert.equal(transition.level, "warning");
  }
});

test("marks a disconnected expired job as orphaned when cancellation is requested", () => {
  const transition = perfJobCancelTransition({
    status: "disconnected",
    lease_expires_at: "2026-08-11T11:59:00.000Z",
  }, Date.parse("2026-08-11T12:00:00.000Z"));
  assert.equal(transition.status, "orphaned");
});
