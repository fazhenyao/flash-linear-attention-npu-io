CREATE TABLE IF NOT EXISTS perf_jobs (
  id TEXT PRIMARY KEY,
  created_by TEXT NOT NULL,
  created_by_username TEXT NOT NULL DEFAULT '',
  idempotency_key TEXT NOT NULL,
  tool TEXT NOT NULL,
  script_id TEXT NOT NULL,
  request_json TEXT NOT NULL,
  status TEXT NOT NULL,
  status_message TEXT NOT NULL DEFAULT '',
  runner_id TEXT,
  attempt_id TEXT,
  lease_token_hash TEXT,
  lease_expires_at TEXT,
  remote_execution_id TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  retry_count INTEGER NOT NULL DEFAULT 0,
  exit_code INTEGER,
  created_at TEXT NOT NULL,
  claimed_at TEXT,
  started_at TEXT,
  finished_at TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE(created_by, idempotency_key)
);

CREATE TABLE IF NOT EXISTS perf_job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  attempt_id TEXT,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perf_results (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL UNIQUE,
  case_id TEXT,
  model_id TEXT,
  snapshot_id TEXT,
  environment_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perf_artifacts (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  object_key TEXT NOT NULL,
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(job_id, object_key)
);

CREATE TABLE IF NOT EXISTS runner_agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  capabilities_json TEXT NOT NULL DEFAULT '{}',
  vpn_connected INTEGER NOT NULL DEFAULT 0,
  npu_reachable INTEGER NOT NULL DEFAULT 0,
  current_jobs INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NOT NULL DEFAULT '',
  last_heartbeat_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_jobs_status_created ON perf_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_perf_jobs_creator_created ON perf_jobs(created_by, created_at);
CREATE INDEX IF NOT EXISTS idx_perf_jobs_runner_status ON perf_jobs(runner_id, status);
CREATE INDEX IF NOT EXISTS idx_perf_job_events_job_created ON perf_job_events(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_perf_artifacts_job ON perf_artifacts(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runner_agents_heartbeat ON runner_agents(last_heartbeat_at);
