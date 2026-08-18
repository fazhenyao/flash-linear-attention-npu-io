import assert from "node:assert/strict";
import test from "node:test";

import {
  RunnerEventHub,
  notifyRunnerJobAvailable,
  runnerEventMessage,
} from "../cloudflare/worker.js";

test("builds a minimal runner wake-up event without job details", () => {
  assert.deepEqual(
    runnerEventMessage("runner-a5", "evt-test", "2026-08-18T08:00:00.000Z"),
    {
      version: 1,
      type: "job_available",
      event_id: "evt-test",
      runner_id: "runner-a5",
      created_at: "2026-08-18T08:00:00.000Z",
    },
  );
});

test("durable object broadcasts notifications to every live runner connection", async () => {
  const messages = [];
  const state = {
    getWebSockets() {
      return [
        { send(message) { messages.push(["one", JSON.parse(message)]); } },
        { send(message) { messages.push(["two", JSON.parse(message)]); } },
      ];
    },
  };
  const hub = new RunnerEventHub(state, {});
  const response = await hub.fetch(new Request("https://runner-events.internal/notify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      runner_id: "runner-a5",
      event_id: "evt-test",
      created_at: "2026-08-18T08:00:00.000Z",
    }),
  }));

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true, delivered: 2 });
  assert.equal(messages.length, 2);
  assert.equal(messages[0][1].type, "job_available");
  assert.equal(messages[1][1].runner_id, "runner-a5");
});

test("targets the durable object for the requested runner", async () => {
  const calls = [];
  const env = {
    RUNNER_EVENTS: {
      idFromName(name) {
        calls.push(["id", name]);
        return `object:${name}`;
      },
      get(id) {
        calls.push(["get", id]);
        return {
          async fetch(request) {
            calls.push(["fetch", request.url, await request.json()]);
            return Response.json({ ok: true, delivered: 1 });
          },
        };
      },
    },
  };

  const result = await notifyRunnerJobAvailable(
    env,
    { target_runner_id: "runner-a5" },
    "perf-job-test",
  );

  assert.equal(result.ok, true);
  assert.equal(result.delivered, 1);
  assert.deepEqual(calls[0], ["id", "runner-a5"]);
  assert.equal(calls[2][2].job_id, "perf-job-test");
});

test("notification is a no-op before the durable object binding is configured", async () => {
  assert.deepEqual(
    await notifyRunnerJobAvailable({}, { target_runner_id: "runner-a5" }, "perf-job-test"),
    { ok: true, enabled: false, delivered: 0, runners: [] },
  );
});
