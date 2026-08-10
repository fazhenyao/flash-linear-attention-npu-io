export function planArtifactStorageCleanup(objects, options) {
  const targetBytes = Number(options.targetBytes);
  const protectedKey = String(options.protectedKey || "");
  const managedPrefix = String(options.managedPrefix || "");
  let usedBytes = objects.reduce((total, object) => total + safeSize(object.size), 0);
  const oldestFirst = objects
    .filter((object) => object.key.startsWith(managedPrefix))
    .sort((left, right) => {
      const leftTime = new Date(left.uploaded || 0).getTime();
      const rightTime = new Date(right.uploaded || 0).getTime();
      return leftTime - rightTime || left.key.localeCompare(right.key);
    });
  const deleted = [];
  for (const object of oldestFirst) {
    if (usedBytes <= targetBytes) break;
    if (object.key === protectedKey) continue;
    deleted.push(object);
    usedBytes -= safeSize(object.size);
  }
  return { usedBytes: Math.max(0, usedBytes), deleted };
}

function safeSize(value) {
  const size = Number(value);
  return Number.isFinite(size) && size > 0 ? size : 0;
}
