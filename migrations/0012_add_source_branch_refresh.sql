ALTER TABLE runner_agents ADD COLUMN source_branches_refresh_id TEXT;
ALTER TABLE runner_agents ADD COLUMN source_branches_requested_at TEXT;
ALTER TABLE runner_agents ADD COLUMN source_branches_repo TEXT;

CREATE INDEX IF NOT EXISTS idx_runner_agents_source_branches_refresh
  ON runner_agents(source_branches_refresh_id);
