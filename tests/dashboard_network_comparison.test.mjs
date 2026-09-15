import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../docs/performance-dashboard.html", import.meta.url), "utf8");
const script = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map((match) => match[1]).join("\n");
// Load the actual page functions without starting network requests or polling.
const source = script.replace(/    init\(\)\.catch\(\(error\) => \{[\s\S]*?\n    \}\);/, "");

function fixture() {
  const nodes = new Map();
  function node(selector) {
    if (!nodes.has(selector)) {
      nodes.set(selector, {
        value: "", checked: true, disabled: false, options: [], events: {}, textContent: "",
        classList: { add() {}, remove() {}, contains() { return false; } },
        addEventListener(name, fn) { this.events[name] = fn; },
        querySelector(child) { return node(`${selector} ${child}`); },
        querySelectorAll() { return []; },
        set innerHTML(value) {
          this.markup = value;
          this.options = [...value.matchAll(/<option value="([^"]*)">/g)].map((match) => ({ value: match[1] }));
          if (this.options.length) this.value = this.options[0].value;
        },
        get innerHTML() { return this.markup || ""; },
      });
    }
    return nodes.get(selector);
  }
  const context = vm.createContext({
    window: {}, Blob, localStorage: { getItem() { return ""; } },
    document: { querySelector: node, querySelectorAll() { return []; }, addEventListener() {} },
    Chart: class { constructor(canvas, config) { this.config = config; } destroy() {} },
  });
  vm.runInContext(source, context);
  const state = vm.runInContext("state", context);
  const op = (backend, time_ms) => ({ operator_id: "shared", backend, time_ms });
  const snap = (id, case_id, operators) => ({
    id, case_id, operators, chip: "A2", prof_tool: "msprof", label: id,
    created_at: id === "old" ? "2026-09-13" : "2026-09-14", prof_source: `PROF_${id}`,
  });
  state.data = { models: [], runs: [], cases: [
    { id: "c1", label: "GDN B=1", example_id: "flash_gated_delta_rule", attributes: { batch: 1, tokens: 128, demo_model: true } },
    { id: "c2", label: "KDA <script>", example_id: "flash_gated_delta_rule", attributes: { batch: 1, tokens: 128, demo_model: true } },
    { id: "hidden", active: false },
  ], snapshots: [
    snap("old", "c1", [op("ascendc", 2), op("ascendc", 3), op("triton", 5)]),
    snap("new", "c1", [op("ascendc", 4)]),
    snap("other", "c2", [op("triton", 8)]),
    snap("hidden", "hidden", [op("ascendc", 1)]),
  ] };
  node("#netCase").value = "c1";
  node("#netSnapshot").value = "old";
  return { state, node, context, run: (code) => vm.runInContext(code, context) };
}

test("dashboard inline JavaScript parses", () => {
  new vm.Script(script);
});

test("same-case historical collections keep separate identity, order, and baseline", () => {
  const f = fixture();
  f.state.networkCompareSnapshots = ["old", "new", "old", "missing", "hidden"];
  const rows = f.run("networkComparisonRows(selectedNetworkSnapshots())");
  assert.deepEqual(Array.from(rows, (row) => row.snap.id), ["old", "new"]);
  assert.equal(rows[0].total, 10);
  assert.equal(rows[0].operators.get("shared::ascendc").time_ms, 5);
  assert.equal(rows[0].operators.get("shared::triton").time_ms, 5);
  assert.equal(rows[1].change, -60);
  assert.equal(rows[1].operators.has("shared::triton"), false);
});

test("history selector, context, and archive all use the selected collection", () => {
  const f = fixture();
  f.run("refreshNetworkSnapshotSelect()");
  assert.equal(f.node("#netSnapshot").value, "old");
  assert.equal(f.node("#netSnapshot").options.length, 2);
  f.run("updateRunContextDisplay('net')");
  assert.match(f.node("#netRunContext .perf-run-context-text").textContent, /PROF_old/);
  f.state.data.runs = [
    { id: "job-new", case_id: "c1", snapshot_id: "new" },
    { id: "job-old", case_id: "c1", snapshot_id: "old" },
  ];
  f.state.jobs = f.state.data.runs.map((run) => ({ id: run.id, status: "succeeded", r2_artifact_count: 1 }));
  assert.equal(f.run("cloudRunForCase('c1', currentSnapshot()).id"), "job-old");
});

test("grouped chart aligns backend-specific operators and uses null for missing data", () => {
  const f = fixture();
  f.state.networkCompareSnapshots = ["old", "new", "other"];
  f.run("renderNetworkTab()");
  const datasets = f.state.charts.networkCompare.config.data.datasets;
  assert.equal(datasets.length, 2);
  assert.deepEqual(Array.from(datasets[0].data), [5, 4, null]);
  assert.deepEqual(Array.from(datasets[1].data), [5, null, 8]);
  assert.equal(f.state.charts.networkCompare.config.data.labels.length, 3);
  assert.match(f.node("#networkComparisonContext").innerHTML, /KDA &lt;script&gt;/);
  assert.match(f.node("#networkComparisonTable").innerHTML, /<td class="perf-op-time"><\/td>/);
});

test("add is idempotent; remove rebaselines; clear restores current single collection", () => {
  const f = fixture();
  f.run("bindEvents(); addNetworkComparison(); addNetworkComparison()");
  assert.equal(f.state.networkCompareSnapshots.length, 1);
  f.node("#netSnapshot").value = "new";
  f.node("#netCompareAddBtn").events.click();
  assert.equal(f.state.networkCompareSnapshots.length, 2);
  f.node("#netCompareList").value = "old";
  f.node("#netCompareRemoveBtn").events.click();
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ["new"]);
  assert.match(f.node("#networkComparisonTable").innerHTML, /采集 1（基准）/);
  f.node("#netCompareClearBtn").events.click();
  assert.equal(f.state.networkCompareSnapshots.length, 0);
  assert.equal(f.state.charts.networkCompare.config.data.datasets.length, 1);
  assert.equal(f.state.charts.networkTotals.config.data.datasets[0].data[0], 4);
});

test("scope filters apply consistently; empty scope and zero baseline do not invent improvements", () => {
  const f = fixture();
  f.state.networkCompareSnapshots = ["old", "new"];
  f.node("#netShowAscendc").checked = false;
  let rows = f.run("networkComparisonRows(selectedNetworkSnapshots())");
  assert.equal(rows[0].total, 5);
  assert.equal(rows[1].total, null);
  assert.equal(rows[1].change, null);
  f.node("#netShowTriton").checked = false;
  f.run("renderNetworkTab()");
  assert.equal(f.state.charts.networkTotals.config.data.datasets[0].data[0], null);
  assert.match(f.node("#networkComparisonTable").innerHTML, /历史逻辑汇总/);
  rows = f.run("networkComparisonRows([{operators:[{operator_id:'x',backend:'ascendc',time_ms:0}]}, {operators:[]}])");
  assert.equal(rows[0].total, 0);
  assert.equal(rows[0].change, null);
  assert.equal(rows[1].change, null);
});

test("candidate filters and data refresh preserve chosen snapshots; deleted cases are pruned", () => {
  const f = fixture();
  f.state.networkCompareSnapshots = ["old", "other"];
  f.state.caseAttrFilters.net = [{ key: "chip", op: "eq", value: "nonexistent" }];
  f.node("#netCase").value = "";
  f.run("renderNetworkTab()");
  assert.equal(f.state.charts.networkCompare.config.data.datasets.length, 2);
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ["old", "other"]);
  f.state.data.cases.find((item) => item.id === "c1").active = false;
  f.run("renderNetworkTab()");
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ["other"]);
});

test("empty initial render waits for data; explicit deselection stays empty", () => {
  const f = fixture();
  f.run("renderComparePick([])");
  assert.equal(f.state.compareOps, null);
  f.run("renderNetworkTab()");
  assert.equal(f.state.compareOps.length, 2);
  f.state.compareOps = [];
  f.run("renderNetworkTab()");
  assert.equal(f.state.charts.networkCompare.config.data.datasets.length, 0);
});

test("reloading data preserves historical collection and chosen operators", async () => {
  const f = fixture();
  f.state.networkCompareSnapshots = ["old", "new"];
  f.state.compareOps = ["shared::triton"];
  f.context.fetch = async (url) => ({ ok: true, json: async () =>
    url.includes("performance-data") ? structuredClone(f.state.data) : { operators: [] } });
  await f.run("loadData()");
  f.run("refreshNetworkSnapshotSelect(); renderNetworkTab()");
  assert.equal(f.node("#netSnapshot").value, "old");
  assert.deepEqual(Array.from(f.state.compareOps), ["shared::triton"]);
  assert.equal(f.state.charts.networkCompare.config.data.labels.length, 2);
});

test("load matching history merges independent cases, orders timestamps, and excludes incompatible inputs", () => {
  const f = fixture();
  f.state.data.cases[1].attributes = { tokens: '128', demo_model: 'true', batch: '1', notes: 'another day' };
  const base = f.state.data.snapshots[0];
  for (const [id, change] of [
    ['shape', { tokens: 256 }], ['mode', { demo_model: false }],
    ['switch', { use_composite_core: true }], ['seed', { seed: 99 }],
  ]) {
    f.state.data.cases.push({ id, example_id: 'flash_gated_delta_rule', attributes: { ...f.state.data.cases[0].attributes, ...change } });
    f.state.data.snapshots.push({ ...base, id, case_id: id });
  }
  f.state.data.snapshots.push({ ...base, id: 'wrong-example', example_id: 'flash_kda' });
  f.state.data.snapshots.push({ ...base, id: 'wrong-profiler', prof_tool: 'msprof_op' });
  f.state.data.snapshots.find((snap) => snap.id === 'other').created_at = '2026-09-12T23:30:00Z';
  f.state.data.snapshots.find((snap) => snap.id === 'old').created_at = '2026-09-13T06:00:00+08:00';
  f.run('loadMatchingNetworkHistory()');
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ['old', 'other', 'new']);
  assert.equal(f.state.networkComparisonTable.headers.length, 8);
  assert.match(f.state.networkComparisonTable.headers[2], /2026-09-13T06:00:00/);
  assert.match(f.state.networkComparisonTable.headers[4], /2026-09-12T23:30:00Z/);
});

test("manual addition rejects a different example or input and keeps existing comparison", () => {
  const f = fixture();
  f.run('addNetworkComparison()');
  f.state.data.cases[1].attributes.tokens = 256;
  f.node('#netCase').value = 'c2';
  f.node('#netSnapshot').value = 'other';
  f.run('addNetworkComparison()');
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ['old']);
  assert.match(f.node('#netCompareHint').textContent, /不同/);
  f.run('loadMatchingNetworkHistory()');
  assert.deepEqual(Array.from(f.state.networkCompareSnapshots), ['other']);
});

test("incomplete legacy metadata never groups unrelated cases, but supports same-case history", () => {
  const f = fixture();
  delete f.state.data.cases[0].example_id;
  delete f.state.data.cases[1].example_id;
  assert.deepEqual(Array.from(f.run('matchingNetworkHistory(currentSnapshot())'), (snap) => snap.id), ['old', 'new']);
});

test("run-specific parameters prevent same-case runs with different inputs from merging", () => {
  const f = fixture();
  f.state.data.runs = [{ snapshot_id: 'new', attributes: { demo_model: false } }];
  assert.deepEqual(Array.from(f.run('matchingNetworkHistory(currentSnapshot())'), (snap) => snap.id), ['old', 'other']);
});

test("CSV exports the full displayed table, independent of chart selection, with numeric missing values", async () => {
  const f = fixture();
  f.run('bindEvents(); loadMatchingNetworkHistory()');
  f.state.compareOps = [];
  f.run('renderNetworkComparison()');
  const table = f.state.networkComparisonTable;
  assert.equal(table.headers.length, 8);
  const timeRow = table.values.find((row) => row[2].includes('shared (triton)'));
  assert.deepEqual(Array.from(timeRow.slice(2)), ['历史逻辑汇总：shared (triton)', 5000, null, null, null, null]);
  assert.equal(table.values[0][7], 8000);
  const csv = f.run('networkComparisonCsv(state.networkComparisonTable)');
  assert.ok(csv.startsWith('\ufeff"Block","SubBlock"'));
  assert.match(csv, /历史逻辑汇总：shared \(ascendc\)/);
  assert.match(csv, /"5000","历史逻辑汇总：shared \(ascendc\)","4000"/);
  assert.ok(!csv.includes('undefined'));
  let download;
  f.context.downloadBlobFile = (name, blob) => { download = { name, blob }; };
  f.node('#netCompareExportBtn').events.click();
  assert.match(download.name, /^network-performance-.*\.csv$/);
  assert.equal(download.blob.type, 'text/csv;charset=utf-8');
  const buffer = new Uint8Array(await download.blob.arrayBuffer());
  assert.deepEqual(Array.from(buffer.slice(0, 3)), [239, 187, 191]);
  assert.equal(await download.blob.text(), csv.slice(1));
});

test("CSV quotes commas, quotes, multiline labels, and neutralizes formulas without changing numeric negatives", () => {
  const f = fixture();
  f.context.exportFixture = { headers: ['名称', '变化'], values: [
    ['a,"b"\nc', -60], ['=HYPERLINK("bad")', null], ['  @SUM(1)', 0],
  ] };
  const csv = f.run('networkComparisonCsv(exportFixture)');
  assert.ok(csv.includes('"a,""b""\nc","-60"\r\n'));
  assert.ok(csv.includes('"\'=HYPERLINK(""bad"")",""'));
  assert.ok(csv.includes('"\'  @SUM(1)","0"'));
});

test("reference layout gives every capture independent name and time columns, preserving source order", () => {
  const f = fixture();
  f.node('#netSummaryStat').value = 'total';
  const stats = (total) => ({ total, avg: total / 2, min: 0, max: total, count: 2 });
  f.state.data.snapshots[0].op_summary = [
    { op_name: 'Exact_old<kernel>', call_count: 2, time_stats: { 'Task Duration(us)': stats(0.5), 'Task Wait Time(us)': stats(0) } },
    { op_name: 'shared_actual_name', call_count: 1, time_stats: { 'Task Duration(us)': stats(3) } },
  ];
  f.state.data.snapshots[1].op_summary = [
    { op_name: 'Exact_new_kernel', call_count: 1, time_stats: { 'Task Duration(us)': stats(2), 'new_time(ms)': stats(1) } },
    { op_name: 'shared_actual_name', call_count: 1, time_stats: { 'Task Duration(us)': stats(4) } },
    { op_name: 'only_new', call_count: 1, time_stats: {} },
  ];
  f.state.networkCompareSnapshots = ['old', 'new'];
  f.node('#netShowAscendc').checked = false;
  f.node('#netShowTriton').checked = false;
  f.run('renderNetworkTab()');
  const table = f.state.networkComparisonTable;
  assert.equal(table.headers.length, 6);
  assert.deepEqual(Array.from(table.values[0]), ['', '', 'Exact_old<kernel>', 0.5, 'Exact_new_kernel', 2]);
  assert.deepEqual(Array.from(table.values[1]), ['', '', 'shared_actual_name', 3, 'shared_actual_name', 4]);
  assert.deepEqual(Array.from(table.values[2]), ['', '', null, null, 'only_new', null]);
  assert.match(f.node('#networkComparisonTable').innerHTML, /Exact_old&lt;kernel&gt;/);
  const csv = f.run('networkComparisonCsv(state.networkComparisonTable)');
  assert.ok(csv.includes('"Exact_old<kernel>","0.5","Exact_new_kernel","2"'));
  f.node('#netSummaryField').value = 'Task Wait Time(us)';
  f.run('renderNetworkComparison()');
  assert.equal(f.state.networkComparisonTable.values[0][3], 0);
  assert.equal(f.state.networkComparisonTable.values[0][5], null);
  f.node('#netSummaryField').value = 'new_time(ms)';
  f.run('renderNetworkComparison()');
  assert.equal(f.state.networkComparisonTable.values[0][5], 1);
  assert.equal(f.state.networkComparisonTable.field, 'new_time(ms)');
});

test("time-stat selector exports exactly displayed stats and never aligns legacy aggregates to raw names", () => {
  const f = fixture();
  f.state.data.snapshots[0].op_summary = [{op_name:'shared',call_count:2,time_stats:{
    'Task Duration(us)':{total:6,avg:3,min:2,max:4,count:2},
    'Task Start Time(us)':{min:100,max:200,count:2},
  }}];
  f.state.networkCompareSnapshots = ['old', 'new'];
  f.run('bindEvents()');
  f.node('#netSummaryStat').value = 'avg';
  f.node('#netSummaryStat').events.change();
  let values = f.state.networkComparisonTable.values;
  assert.deepEqual(Array.from(values[0].slice(2)), ['shared', 3, '历史逻辑汇总：shared (ascendc)', null]);
  f.node('#netSummaryStat').value = 'all';
  f.node('#netSummaryStat').events.change();
  values = f.state.networkComparisonTable.values;
  assert.deepEqual(Array.from(values[0].slice(2)), ['shared', 6, 3, 2, 4, '历史逻辑汇总：shared (ascendc)', 4000, null, null, null]);
  f.node('#netSummaryField').value = 'Task Start Time(us)';
  f.node('#netSummaryField').events.change();
  assert.deepEqual(Array.from(f.state.networkComparisonTable.values[0].slice(2, 7)), ['shared', null, null, 100, 200]);
  const csv = f.run('networkComparisonCsv(state.networkComparisonTable)');
  assert.ok(csv.includes('"shared","","","100","200"'));
});

test("Block and SubBlock groups only align explicitly recorded categories and CSV remains rectangular", () => {
  const f = fixture();
  f.node('#netSummaryStat').value = 'total';
  const op = (name, block, sub_block, total) => ({op_name:name,block,sub_block,time_stats:{'Task Duration(us)':{total}}});
  f.state.data.snapshots[0].op_summary = [op('a','Prefill','Attention',1),op('b','Prefill','MLP',2)];
  f.state.data.snapshots[1].op_summary = [op('c','Prefill','MLP',3),op('d','Decode','Attention',4)];
  f.state.networkCompareSnapshots = ['old','new'];
  f.run('renderNetworkComparison()');
  const table = f.state.networkComparisonTable;
  assert.deepEqual(Array.from(table.values, row=>Array.from(row)), [
    ['Prefill','Attention','a',1,null,null],
    ['Prefill','MLP','b',2,'c',3],
    ['Decode','Attention',null,null,'d',4],
  ]);
  assert.ok(table.values.every(row=>row.length===table.headers.length));
  assert.match(f.node('#networkComparisonTable').innerHTML, /colspan="2"/);
});

test("default reference layout preserves every repeated execution and mismatched lists without name alignment", () => {
  const f = fixture();
  f.state.data.snapshots[0].op_summary_details = {
    fields:['Task Duration(us)','Task Start Time(us)'], names:[{op_name:'Repeat<op>'},{op_name:'middle'}],
    rows:[[0,0.125,100],[1,2,101],[0,0.375,102]],
  };
  f.state.data.snapshots[1].op_summary_details = {
    fields:['Task Duration(us)'], names:[{op_name:'different'}], rows:[[0,0],[0,null]],
  };
  f.state.networkCompareSnapshots = ['old','new'];
  f.run('renderNetworkComparison()');
  assert.deepEqual(Array.from(f.state.networkComparisonTable.values, row=>Array.from(row)), [
    ['','','Repeat<op>',0.125,'different',0],
    ['','','middle',2,'different',null],
    ['','','Repeat<op>',0.375,null,null],
  ]);
  assert.match(f.node('#networkComparisonTable').innerHTML, /Repeat&lt;op&gt;/);
  const csv = f.run('networkComparisonCsv(state.networkComparisonTable)');
  assert.equal(csv.split('"Repeat<op>"').length - 1, 2);
  assert.ok(csv.includes('"Repeat<op>","0.375","",""'));
  f.node('#netSummaryField').value = 'Task Start Time(us)';
  f.run('renderNetworkComparison()');
  assert.equal(f.state.networkComparisonTable.values[2][3], 102);
});

test("old named summaries are visibly distinguished from per-execution records", () => {
  const f = fixture();
  f.state.data.snapshots[0].op_summary = [{op_name:'repeat',time_stats:{'Task Duration(us)':{total:2,avg:1}}}];
  f.run('renderNetworkComparison()');
  assert.equal(f.state.networkComparisonTable.values[0][2], 'repeat【同名汇总，缺逐次明细】');
  assert.equal(f.state.networkComparisonTable.values[0][3], 2);
});
