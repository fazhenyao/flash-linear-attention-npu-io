-- The A5 Relay retained a successful acknowledgement for this exact attempt.
-- If a late heartbeat revived it, confirm the matching stored result before
-- repairing the queue row. Never touch a retry or a subsequently canceled job.
UPDATE perf_jobs SET status = 'succeeded',
  status_message = 'msprof（整网） 执行并导入：PROF_000001_20260818162301373_03627029QFKAIFMO',
  lease_token_hash = NULL, lease_expires_at = NULL,
  updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id = 'perf-job-e3bf888e-cd75-4723-a576-e7d642f35d05'
  AND attempt_id = 'attempt-4dae6b69-d180-487e-915a-e6392239d61f'
  AND status IN ('running', 'disconnected') AND exit_code = 0 AND finished_at IS NOT NULL
  AND EXISTS (
    SELECT 1 FROM perf_results WHERE job_id = perf_jobs.id
      AND snapshot_id = 'snap-prof_000001_20260818162301373_03627029qfkaifmo'
  );

-- Repair historical display copies from the authoritative queue. Keep runs
-- without a queue record untouched; elapsed time alone does not prove failure.
UPDATE project_meta
SET value = json_set(value, '$.runs', json((
  SELECT json_group_array(json(item)) FROM (
    SELECT CASE WHEN job.id IS NULL THEN run.value ELSE json_set(run.value,
      '$.status', CASE job.status WHEN 'succeeded' THEN 'done' WHEN 'orphaned' THEN 'failed' ELSE job.status END,
      '$.message', job.status_message,
      '$.started_at', job.started_at,
      '$.finished_at', job.finished_at
    ) END AS item
    FROM json_each(project_meta.value, '$.runs') AS run
    LEFT JOIN perf_jobs AS job ON job.id = json_extract(run.value, '$.id')
    ORDER BY CAST(run.key AS INTEGER)
  )
)), '$.version', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
WHERE key = 'perfData';
