import assert from "node:assert/strict";
import test from "node:test";
import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { projectPerfJobRun, mergePerfJobCompletion, sweepExpiredPerfLeases, runnerJobHeartbeat,
  runnerJobStarted, runnerJobComplete, runnerJobFail, cancelPerfJob } from "../cloudflare/worker.js";

function fixture() {
  const db = new DatabaseSync(":memory:");
  db.exec("CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
  db.exec(readFileSync(new URL("../migrations/0009_add_perf_job_queue.sql", import.meta.url), "utf8"));
  let heldReads = null;
  const prepare = (sql) => ({ bind(...params) {
    const stmt = db.prepare(sql);
    return {
      async run() { return stmt.run(...params); },
      runSync() { return stmt.run(...params); },
      async first() {
        const result = stmt.get(...params) || null;
        if (heldReads && /SELECT.*FROM project_meta/i.test(sql)) {
          return new Promise((resolve) => {
            heldReads.push(() => resolve(result));
            if (heldReads.length === 2) {
              const ready = heldReads;
              heldReads = null;
              ready.forEach((release) => release());
            }
          });
        }
        return result;
      },
      async all() { return { results: stmt.all(...params) }; },
    };
  } });
  const env = { DB: { prepare, async batch(statements) {
    db.exec("BEGIN");
    try {
      const results = [];
      for (const statement of statements) results.push(statement.runSync());
      db.exec("COMMIT");
      return results;
    } catch (error) { db.exec("ROLLBACK"); throw error; }
  } } };
  function job(id, status = "running") {
    db.prepare(`INSERT INTO perf_jobs (id, created_by, idempotency_key, tool, script_id, request_json,
      status, created_at, updated_at, lease_expires_at) VALUES (?, 'user', ?, 'msprof', 'gdn', '{}', ?, '2026-08-18', '2026-08-18', '2026-08-19')`)
      .run(id, id, status);
    return db.prepare("SELECT * FROM perf_jobs WHERE id = ?").get(id);
  }
  const data = () => JSON.parse(db.prepare("SELECT value FROM project_meta WHERE key = 'perfData'").get().value);
  return { db, env, job, data, holdConcurrentPerfReads() { heldReads = []; } };
}

test("a completion and another task's projection preserve both runs and result collections", async () => {
  const { db, env, job, data, holdConcurrentPerfReads } = fixture();
  const a = job("a", "succeeded");
  const b = job("b");
  await projectPerfJobRun(env, a);
  // Reproduce both handlers reading the same JSON before either writes it.
  // Atomic SQL updates do not make these vulnerable application-side reads.
  holdConcurrentPerfReads();
  await Promise.all([mergePerfJobCompletion(env, "a", { perf_data: {
    cases: [{ id: "case-a", label: "A" }], snapshots: [{ id: "snap-a", case_id: "case-a" }],
  } }, {}, { id: "snap-a", case_id: "case-a" }), projectPerfJobRun(env, b)]);
  assert.equal(data().runs.find((run) => run.id === "a").status, "done");
  assert.equal(data().runs.find((run) => run.id === "b").status, "running");
  assert.equal(data().snapshots[0].id, "snap-a");
  db.prepare("UPDATE perf_jobs SET status = 'succeeded', updated_at = '2026-08-20' WHERE id = 'b'").run();
  await mergePerfJobCompletion(env, "b", { perf_data: {
    cases: [{ id: "case-b" }], snapshots: [{ id: "snap-b" }],
  } }, {}, { id: "snap-b" });
  assert.deepEqual(data().cases.map((item) => item.id), ["case-a", "case-b"]);
  assert.deepEqual(data().snapshots.map((item) => item.id), ["snap-a", "snap-b"]);
  // A delayed started callback must not revert this same job to running.
  await projectPerfJobRun(env, b);
  assert.equal(data().runs.find((run) => run.id === "b").status, "done");
  db.close();
});

test("a heartbeat that was in flight during completion cannot revive the finished job", async () => {
  const { db, env, job, data } = fixture();
  const stale = job("a");
  db.prepare("UPDATE perf_jobs SET status = 'succeeded' WHERE id = 'a'").run();
  await runnerJobHeartbeat(env, stale, {});
  assert.equal(db.prepare("SELECT status FROM perf_jobs WHERE id = 'a'").get().status, "succeeded");
  const current = job("b");
  await runnerJobHeartbeat(env, current, {});
  // Routine heartbeats must not suppress the delayed started projection.
  await projectPerfJobRun(env, current);
  assert.equal(data().runs[0].status, "running");
  db.close();
});

test("completion replay is idempotent and run projection preserves result fields and explicit nulls", async () => {
  const { db, env, job, data } = fixture();
  const row = job("a", "succeeded");
  const payload = { perf_data: { cases: [{ id: "c", label: "updated" }] }, command: "python example.py" };
  await mergePerfJobCompletion(env, row.id, payload, {}, { id: "s", case_id: "c" });
  await mergePerfJobCompletion(env, row.id, payload, {}, { id: "s", case_id: "c" });
  await projectPerfJobRun(env, row);
  assert.equal(data().runs.length, 1);
  assert.equal(data().snapshots.length, 1);
  assert.equal(data().cases.length, 1);
  assert.equal(data().runs[0].command, "python example.py");
  assert.equal(data().runs[0].finished_at, null);
  db.close();
});

test("expired leases project disconnected and requeued status to the dashboard", async () => {
  const { db, env, job, data } = fixture();
  await projectPerfJobRun(env, job("running"));
  await projectPerfJobRun(env, job("claimed", "claimed"));
  await sweepExpiredPerfLeases(env);
  assert.equal(data().runs.find((run) => run.id === "running").status, "disconnected");
  assert.equal(data().runs.find((run) => run.id === "claimed").status, "queued");
  db.close();
});

test("repair migration uses authoritative status without inventing outcomes for unknown runs", async () => {
  const { db, env, job, data } = fixture();
  await projectPerfJobRun(env, job("a"));
  db.prepare("UPDATE perf_jobs SET status = 'succeeded', finished_at = '2026-08-18T08:24:29Z' WHERE id = 'a'").run();
  const before = data();
  before.runs.push({ id: "legacy", status: "running" });
  db.prepare("UPDATE project_meta SET value = ? WHERE key = 'perfData'").run(JSON.stringify(before));
  db.exec(readFileSync(new URL("../migrations/0013_reconcile_perf_run_status.sql", import.meta.url), "utf8"));
  assert.equal(data().runs[0].status, "done");
  assert.equal(data().runs[0].finished_at, "2026-08-18T08:24:29Z");
  assert.deepEqual(data().runs[1], { id: "legacy", status: "running" });
  db.close();
});

test("dashboard polling follows live queue status and ignores stale or fallback snapshots", () => {
  const html = readFileSync(new URL("../docs/performance-dashboard.html", import.meta.url), "utf8");
  const fn = html.slice(html.indexOf("    function hasPendingPerfRuns()"), html.indexOf("    function maybeStartPerfPolling()"));
  function pending(jobs, dataSource, token = "token", worker = "https://worker") {
    return vm.runInNewContext(`${fn}; hasPendingPerfRuns()`, {
      WORKER_API_BASE: worker,
      state: { jobs, token, dataSource, data: { runs: [{ id: "old", status: "running" }] } },
    });
  }
  assert.equal(pending([{ id: "old", status: "succeeded" }], "worker"), false);
  assert.equal(pending([], "local-fallback"), false);
  assert.equal(pending([], "local-fallback", ""), false);
  assert.equal(pending([{ id: "active", status: "running" }], "worker"), true);
  assert.equal(pending([], "local-api", "", ""), true);
});

test("historical A5 repair requires the acknowledged attempt and a stored successful result", async () => {
  const { db, env, job, data } = fixture();
  const id = "perf-job-e3bf888e-cd75-4723-a576-e7d642f35d05";
  await projectPerfJobRun(env, job(id));
  const sql = readFileSync(new URL("../migrations/0013_reconcile_perf_run_status.sql", import.meta.url), "utf8");
  db.exec(sql);
  assert.equal(data().runs[0].status, "running");
  db.prepare(`UPDATE perf_jobs SET attempt_id = 'attempt-4dae6b69-d180-487e-915a-e6392239d61f',
    exit_code = 0, finished_at = '2026-08-18T08:24:29Z' WHERE id = ?`).run(id);
  db.prepare(`INSERT INTO perf_results (id, job_id, snapshot_id, created_at) VALUES ('r', ?,
    'snap-prof_000001_20260818162301373_03627029qfkaifmo', '2026-08-18T08:24:29Z')`).run(id);
  db.exec(sql);
  assert.equal(data().runs[0].status, "done");
  db.prepare("UPDATE perf_jobs SET status = 'running', attempt_id = 'new-attempt' WHERE id = ?").run(id);
  db.exec(sql);
  assert.equal(data().runs[0].status, "running");
  db.close();
});

test("cancellation wins against in-flight started and completion reports", async () => {
  const { db, env, job, data } = fixture();
  const stale = job("cancel-race");
  db.prepare("UPDATE perf_jobs SET status='cancel_requested', cancel_requested=1 WHERE id=?").run(stale.id);
  const started = await runnerJobStarted(env, stale, {});
  assert.equal(started.cancel_requested, true);
  assert.equal(started.job.status, "cancel_requested");
  const result = await runnerJobComplete(env, stale, { result: {} });
  assert.equal(result.job.status, "canceled");
  assert.equal(data().runs[0].status, "canceled");
  db.close();
});

test("late failure and cancel cannot revive a succeeded task", async () => {
  const { db, env, job } = fixture();
  const stale = job("complete-race");
  db.prepare("UPDATE perf_jobs SET status='succeeded' WHERE id=?").run(stale.id);
  await runnerJobFail(env, stale, { canceled: true });
  const result = await cancelPerfJob(env, stale.id, { id: 'user', role: 'admin' });
  assert.equal(result.job.status, "succeeded");
  db.close();
});

test("cancel retries a concurrent claim instead of silently losing the request", async () => {
  const { db, env, job } = fixture();
  const row = job("claim-race", "queued");
  const prepare = env.DB.prepare;
  let raced = false;
  env.DB.prepare = (sql) => {
    if (!raced && sql.includes("cancel_requested = 1, finished_at")) {
      raced = true;
      db.prepare("UPDATE perf_jobs SET status='claimed', attempt_id='new-attempt' WHERE id=?").run(row.id);
    }
    return prepare(sql);
  };
  const result = await cancelPerfJob(env, row.id, { id: 'user', role: 'admin' });
  assert.equal(result.job.status, "cancel_requested");
  assert.equal(result.job.attempt_id, "new-attempt");
  assert.equal(result.job.cancel_requested, true);
  db.close();
});
