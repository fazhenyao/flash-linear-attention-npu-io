import assert from "node:assert/strict";
import test from "node:test";

import { planArtifactStorageCleanup } from "../cloudflare/artifact_storage.js";

const options = {
  targetBytes: 9_000_000_000,
  managedPrefix: "perf-artifacts/",
};

test("deletes the oldest managed artifacts until usage is below the target", () => {
  const objects = [
    { key: "perf-artifacts/job-new/artifact", size: 4_000_000_000, uploaded: "2026-08-10T00:00:00Z" },
    { key: "perf-artifacts/job-old/artifact", size: 3_000_000_000, uploaded: "2026-08-01T00:00:00Z" },
    { key: "perf-artifacts/job-middle/artifact", size: 3_000_000_000, uploaded: "2026-08-05T00:00:00Z" },
  ];

  const plan = planArtifactStorageCleanup(objects, options);

  assert.equal(plan.usedBytes, 7_000_000_000);
  assert.deepEqual(plan.deleted.map((object) => object.key), ["perf-artifacts/job-old/artifact"]);
});

test("counts unmanaged objects but never deletes them", () => {
  const objects = [
    { key: "manual/keep", size: 8_000_000_000, uploaded: "2026-07-01T00:00:00Z" },
    { key: "perf-artifacts/job-old/artifact", size: 2_000_000_000, uploaded: "2026-08-01T00:00:00Z" },
  ];

  const plan = planArtifactStorageCleanup(objects, options);

  assert.equal(plan.usedBytes, 8_000_000_000);
  assert.deepEqual(plan.deleted.map((object) => object.key), ["perf-artifacts/job-old/artifact"]);
});

test("protects the newly completed artifact from cleanup", () => {
  const protectedKey = "perf-artifacts/job-new/artifact";
  const objects = [
    { key: protectedKey, size: 6_000_000_000, uploaded: "2026-08-10T00:00:00Z" },
    { key: "perf-artifacts/job-old/artifact", size: 5_000_000_000, uploaded: "2026-08-01T00:00:00Z" },
  ];

  const plan = planArtifactStorageCleanup(objects, { ...options, protectedKey });

  assert.equal(plan.usedBytes, 6_000_000_000);
  assert.deepEqual(plan.deleted.map((object) => object.key), ["perf-artifacts/job-old/artifact"]);
});

test("cannot reclaim capacity from unmanaged objects", () => {
  const objects = [
    { key: "manual/keep", size: 10_000_000_000, uploaded: "2026-07-01T00:00:00Z" },
  ];

  const plan = planArtifactStorageCleanup(objects, options);

  assert.equal(plan.usedBytes, 10_000_000_000);
  assert.deepEqual(plan.deleted, []);
});
