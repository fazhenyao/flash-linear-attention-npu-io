// Mutate the current D1 value inside SQL, so concurrent jobs cannot replace one
// another's updates with an earlier copy of the whole performance document.
export function preparePerfCollectionMerge(env, collection, items, timestamp) {
  if (!["models", "cases", "snapshots"].includes(collection)) throw new Error("invalid performance collection");
  const unique = [...new Map(items.filter((item) => item?.id).map((item) => [item.id, item])).values()];
  const path = `$.${collection}`;
  return env.DB.prepare(`
    WITH incoming AS (
      SELECT value AS item, json_extract(value, '$.id') AS id FROM json_each(?)
    )
    UPDATE project_meta SET value = json_set(value, ?, json((
      SELECT json_group_array(json(item)) FROM (
        SELECT COALESCE((SELECT item FROM incoming WHERE id = json_extract(old.value, '$.id')), old.value) AS item
        FROM json_each(project_meta.value, ?) AS old
        UNION ALL
        SELECT item FROM incoming WHERE NOT EXISTS (
          SELECT 1 FROM json_each(project_meta.value, ?) AS old
          WHERE json_extract(old.value, '$.id') = incoming.id
        )
      )
    )), '$.version', ?) WHERE key = 'perfData'
  `).bind(JSON.stringify(unique), path, path, path, timestamp);
}

export function preparePerfRunProjection(env, row, run, timestamp) {
  const fields = Object.entries(run).filter(([, value]) => value !== undefined);
  const setters = fields.map(() => "?, json(?)").join(", ");
  return env.DB.prepare(`
    UPDATE project_meta SET value = json_set(value,
      '$.runs[' || COALESCE(
        (SELECT old.key FROM json_each(project_meta.value, '$.runs') AS old WHERE json_extract(old.value, '$.id') = ?),
        json_array_length(value, '$.runs'), 0
      ) || ']',
      json_set(COALESCE(
        (SELECT old.value FROM json_each(project_meta.value, '$.runs') AS old WHERE json_extract(old.value, '$.id') = ?), '{}'
      ), ${setters}), '$.version', ?)
    WHERE key = 'perfData' AND EXISTS (
      SELECT 1 FROM perf_jobs WHERE id = ? AND status = ? AND attempt_id IS ? AND retry_count = ?
    )
  `).bind(
    row.id, row.id, ...fields.flatMap(([key, value]) => [`$.${key}`, JSON.stringify(value)]),
    timestamp, row.id, row.status, row.attempt_id || null, row.retry_count || 0,
  );
}
