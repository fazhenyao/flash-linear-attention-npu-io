import assert from "node:assert/strict";
import test from "node:test";

import { normalizePerfExecutionEnvironment, runnerCanExecute } from "../cloudflare/worker.js";

test("normalizes a controlled source build environment", () => {
  assert.deepEqual(normalizePerfExecutionEnvironment({
    cann_path: "/home/user/cann/ascend-toolkit/",
    conda_env: "f30077529",
    source_repo: "/home/user/flash-linear-attention-npu/",
    rebuild: true,
    branch: "feature/a5-build",
  }), {
    cann_path: "/home/user/cann/ascend-toolkit",
    conda_env: "f30077529",
    source_repo: "/home/user/flash-linear-attention-npu",
    rebuild: true,
    branch: "feature/a5-build",
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
