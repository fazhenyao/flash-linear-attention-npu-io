CREATE TABLE IF NOT EXISTS perf_artifact_upload_reservations (
  artifact_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  object_key TEXT NOT NULL UNIQUE,
  size_bytes INTEGER NOT NULL,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_artifact_reservations_expiry
  ON perf_artifact_upload_reservations(expires_at);
