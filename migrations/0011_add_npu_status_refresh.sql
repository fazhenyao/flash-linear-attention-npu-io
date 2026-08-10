ALTER TABLE runner_agents ADD COLUMN npu_status_refresh_id TEXT;
ALTER TABLE runner_agents ADD COLUMN npu_status_refresh_requested_at TEXT;

CREATE INDEX IF NOT EXISTS idx_runner_agents_npu_status_refresh
  ON runner_agents(npu_status_refresh_id);
