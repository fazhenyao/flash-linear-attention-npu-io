import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const context = vm.createContext({ TextEncoder });
vm.runInContext(readFileSync(new URL("../docs/perf-xlsx-export.js", import.meta.url), "utf8"), context);
const exporter = context.PerfXlsxExport;
const plain = (v) => JSON.parse(JSON.stringify(v));
const capture = (changes = {}) => ({
  id: "r1", caseId: "c1", caseKey: "same-input", example: "chunk_gdn", platform: "A5PR", version: "26.9.1",
  time: "2026-10-13T01:00:00Z", status: "done", attributes: { layout: "BSND", tokens: 128 },
  source: "PROF_1", snapshot: { op_summary_details: {
    fields: ["Task Start Time(us)", "Task Duration(us)"], names: [
      { op_name: "aclnnChunkGatedDeltaRuleFwd_Cast_Cast", input_shapes: "1,128,16,128" },
      { op_name: "aclnnChunkGatedDeltaRuleFwd_ChunkFwdH_ChunkFwdH", input_shapes: "1,16,128,128;1,32,128,128", input_formats: "ND;ND" },
      { op_name: "aclnnChunkGatedDeltaRuleBwd_ChunkGdnBwdIntra_ChunkGdnBwdIntra" },
    ], rows: [[0, 100, 0.125], [1, 101, 1.5], [0, 102, 0], [2, 103, 2.75]],
  } }, ...changes,
});
const values = (sheet) => sheet.rows.flatMap((r) => r.filter(Boolean).map((c) => c.v));
const allCells = (sheet) => sheet.rows.flatMap((r) => r.filter(Boolean));

test("case sheets preserve duplicates, execution order, actual shape and per-direction totals", () => {
  const workbook = exporter.buildWorkbook([capture(), capture({ id: "r2", platform: "A5DT" }), capture({ id: "r3", caseKey: "different-input" })]);
  assert.deepEqual(plain(workbook.sheets.map((s) => s.name)), ["case0", "case1", "来源明细"]);
  const sheet = workbook.sheets[0];
  assert.deepEqual(plain(sheet.columns), [20, 13, 83, 17, 83, 17]);
  assert.deepEqual(plain(sheet.rows.filter((r) => r[2]?.v?.startsWith?.("aclnn")).map((r) => r[2].v)), [
    "aclnnChunkGatedDeltaRuleFwd_Cast_Cast", "aclnnChunkGatedDeltaRuleFwd_ChunkFwdH_ChunkFwdH",
    "aclnnChunkGatedDeltaRuleFwd_Cast_Cast", "aclnnChunkGatedDeltaRuleBwd_ChunkGdnBwdIntra_ChunkGdnBwdIntra",
  ]);
  assert.deepEqual(plain(allCells(sheet).filter((c) => c.f).map((c) => c.v)), [1.625, 1.625, 2.75, 2.75]);
  assert.equal(values(sheet).includes("有性能数据"), false);
  assert.match(sheet.rows[2][2].v, /1,16,128,128;1,32,128,128/);
  assert.match(sheet.rows[2][2].v, /轴 layout 未记录；存储格式：ND;ND/);
  assert.equal(workbook.sheets[2].rows.length, 4);
});

test("explicit case grouping aligns disjoint operators by platform/version without merging repeated captures", () => {
  const kda = capture({ id: "kda", example: "chunk_kda", caseKey: "kda-input", sheetGroup: "case0", snapshot: {
    op_summary_details: { fields: ["Task Duration(us)"], names: [{ op_name: "aclnnChunkKdaFwd_ChunkKdaFwd" }], rows: [[0, 3]] },
  } });
  const workbook = exporter.buildWorkbook([capture({ sheetGroup: "case0" }), kda, capture({ id: "again", sheetGroup: "case0" })]);
  const sheet = workbook.sheets[0];
  assert.equal(workbook.sheets.length, 2);
  assert.equal(sheet.columns.length, 6);
  const kdaRow = sheet.rows.find((r) => r[2]?.v === "aclnnChunkKdaFwd_ChunkKdaFwd");
  assert.equal(kdaRow[3].v, 3);
  assert.equal(kdaRow[4], undefined);
  assert.equal(workbook.sheets[1].rows[1][1].v, workbook.sheets[1].rows[2][1].v);
  assert.notEqual(workbook.sheets[1].rows[1][1].v, workbook.sheets[1].rows[3][1].v);
});

test("explicit case page numbers are retained and cannot collide with automatic page names", () => {
  const workbook = exporter.buildWorkbook([capture(), capture({ id: "second", sheetGroup: "case0" }), capture({ id: "third", sheetGroup: "CASE2" })]);
  assert.deepEqual(plain(workbook.sheets.map((s) => s.name)), ["case1", "case0", "case2", "来源明细"]);
});

test("missing, zero, failed and unsupported data cannot become successful partial totals", () => {
  const missing = capture(); missing.snapshot.op_summary_details.rows[1][2] = null;
  const workbook = exporter.buildWorkbook([missing, capture({ id: "failed", status: "failed" }), capture({ id: "unsupported", status: "unsupported", snapshot: {} })]);
  const sheet = workbook.sheets[0];
  assert.equal(sheet.rows.find((r) => r[2]?.v === "aclnnChunkGatedDeltaRuleFwd_ChunkFwdH_ChunkFwdH")[3].v, null);
  assert.equal(allCells(sheet).filter((c) => c.f).length, 1); // Only the complete successful backward section.
  assert.ok(values(sheet).includes("缺少耗时"));
  assert.ok(values(sheet).includes("运行失败"));
  assert.ok(values(sheet).includes("不支持"));
  assert.ok(allCells(sheet).some((c) => c.v === 0));
});

test("only Task Duration contributes to times, with unit conversion and invalid-value handling", () => {
  const r = capture({ snapshot: { op_summary_details: { fields: ["Task Start Time(us)", "Task Duration(ns)"],
    names: [{ op_name: "x" }], rows: [[0, 1000000, 125], [0, 2000000, ""], [0, 3, -1], [0, 4, 0]] } } });
  assert.deepEqual(plain(exporter.executionTasks(r).tasks.map((t) => t.time)), [0.125, null, null, 0]);
  assert.throws(() => exporter.executionTasks(capture({ snapshot: { op_summary_details: { fields: [], names: [], rows: [[5]] } } })), /无效/);
});

test("old summaries remain visibly distinct and never fabricate device names or sample means", () => {
  const named = exporter.executionTasks(capture({ snapshot: { op_summary: [{ op_name: "x", time_stats: { "Task Duration(ms)": { total: 1.5, avg: .5 } } }] } }));
  assert.equal(named.tasks[0].time, 1500);
  assert.equal(named.tasks[0].op_name, "x【同名汇总】");
  const logical = exporter.executionTasks(capture({ snapshot: { operators: [{ operator_id: "kda", time_ms: .125 }] } }));
  assert.equal(logical.tasks[0].time, 125);
  assert.equal(logical.tasks[0].op_name, "历史逻辑汇总：kda");
  assert.match(logical.kind, /缺少原始小算子名称/);
  assert.throws(() => exporter.buildWorkbook([]), /至少/);
});

function storedZipEntries(bytes) {
  const entries = new Map(); const data = Buffer.from(bytes);
  let offset = 0;
  while (data.readUInt32LE(offset) === 0x04034b50) {
    assert.equal(data.readUInt16LE(offset + 8), 0);
    const size = data.readUInt32LE(offset + 18), nameSize = data.readUInt16LE(offset + 26), extra = data.readUInt16LE(offset + 28);
    const name = data.subarray(offset + 30, offset + 30 + nameSize).toString("utf8");
    const start = offset + 30 + nameSize + extra;
    entries.set(name, data.subarray(start, start + size).toString("utf8")); offset = start + size;
  }
  assert.equal(data.readUInt32LE(offset), 0x02014b50);
  assert.equal(data.readUInt32LE(data.length - 22), 0x06054b50);
  assert.equal(data.readUInt16LE(data.length - 12), entries.size);
  return entries;
}

test("XLSX package contains standalone workbook relationships, cached formulas and template styles", () => {
  const files = storedZipEntries(exporter.writeXlsx(exporter.buildWorkbook([capture()])));
  assert.ok(files.has("[Content_Types].xml")); assert.ok(files.has("_rels/.rels"));
  assert.match(files.get("xl/workbook.xml"), /name="来源明细"/);
  assert.match(files.get("xl/_rels/workbook.xml.rels"), /Target="worksheets\/sheet2.xml"/);
  const sheet = files.get("xl/worksheets/sheet1.xml");
  assert.match(sheet, /topLeftCell="C3"/);
  assert.match(sheet, /<f>ROUND\(SUM\(D6:D8\),6\)<\/f><v>1.625<\/v>/);
  assert.match(sheet, /<mergeCells/);
  assert.match(files.get("xl/styles.xml"), /Microsoft YaHei/);
  assert.match(files.get("xl/styles.xml"), /FFD0CECE/);
  assert.match(files.get("xl/styles.xml"), /formatCode="0.000"/);
});

test("imported text is XML-escaped and remains literal even when it resembles a formula", () => {
  const record = capture({ source: '=HYPERLINK("https://example.com")', attributes: { note: "<unsafe>&\"\u0001" } });
  record.snapshot.op_summary_details.names[0].op_name = "=1+1";
  const files = storedZipEntries(exporter.writeXlsx(exporter.buildWorkbook([record])));
  assert.match(files.get("xl/worksheets/sheet2.xml"), /t="inlineStr"><is><t xml:space="preserve">=HYPERLINK/);
  assert.ok(files.get("xl/worksheets/sheet1.xml").includes('&lt;unsafe&gt;&amp;\\&quot;'));
  assert.equal(files.get("xl/worksheets/sheet1.xml").includes("\u0001"), false);
});
