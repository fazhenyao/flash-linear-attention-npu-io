const REPO = { owner: "fazhenyao", name: "flash-linear-attention-npu-io", branch: "main" };
const API_ROOT = `https://api.github.com/repos/${REPO.owner}/${REPO.name}`;
const WORKER_API_BASE = String(window.FLASH_IO_API_BASE || localStorage.getItem("flashWorkerApiBase") || "").replace(/\/+$/, "");
const DATA_PATHS = {
  state: "data/project-state.json",
  audit: "data/audit-log.jsonl",
  pageState: "docs/project-state.json",
  pageAudit: "docs/audit-log.jsonl",
};

const CHINA_WORK_CALENDARS = {
  2026: buildChinaWorkCalendar([
    ["元旦", "2026-01-01", "2026-01-03"],
    ["春节", "2026-02-15", "2026-02-23"],
    ["清明节", "2026-04-04", "2026-04-06"],
    ["劳动节", "2026-05-01", "2026-05-05"],
    ["端午节", "2026-06-19", "2026-06-21"],
    ["中秋节", "2026-09-25", "2026-09-27"],
    ["国庆节", "2026-10-01", "2026-10-07"],
  ], [
    ["2026-01-04", "元旦调休上班"],
    ["2026-02-14", "春节调休上班"],
    ["2026-02-28", "春节调休上班"],
    ["2026-05-09", "劳动节调休上班"],
    ["2026-09-20", "国庆节调休上班"],
    ["2026-10-10", "国庆节调休上班"],
  ]),
};

const OPERATOR_RULES = [
  { id: "chunk_gated_delta_rule_fwd_h", label: "chunk_gated_delta_rule_fwd_h", aliases: ["chunk_gated_delta_rule_fwd_h", "fwd_h"] },
  { id: "chunk_fwd_o", label: "chunk_fwd_o", aliases: ["chunk_fwd_o", "fwd_o"] },
  { id: "recompute_wu_fwd", label: "recompute_wu_fwd", aliases: ["recompute_wu_fwd", "recompute_w_u", "recompute_wu", "recompute"] },
  { id: "chunk_bwd_dv_local", label: "chunk_bwd_dv_local", aliases: ["chunk_bwd_dv_local", "chunk_dv_local", "dv_local"] },
  { id: "chunk_bwd_dqkwg", label: "chunk_bwd_dqkwg", aliases: ["chunk_bwd_dqkwg", "dqkwg"] },
  { id: "chunk_gated_delta_rule_bwd_dhu", label: "chunk_gated_delta_rule_bwd_dhu", aliases: ["chunk_gated_delta_rule_bwd_dhu", "dhu"] },
  { id: "prepare_wy_repr_bwd_da", label: "prepare_wy_repr_bwd_da", aliases: ["prepare_wy_repr_bwd_da", "prepare_wy_bwd_da"] },
  { id: "prepare_wy_repr_bwd_full", label: "prepare_wy_repr_bwd_full", aliases: ["prepare_wy_repr_bwd_full", "prepare_wy_bwd_full"] },
  { id: "causal_conv1d_fwd", label: "causal_conv1d_fwd", aliases: ["causal_conv1d_fwd", "causal_conv1d TND", "TND 转 NTD"] },
  { id: "causal_conv1d_bwd", label: "causal_conv1d_bwd", aliases: ["causal_conv1d_bwd", "causal_conv1d bwd"] },
  { id: "solve_tril", label: "solve_tril", aliases: ["solve_tril", "solve_tri"] },
  { id: "kimi_delta_attention_triton", label: "kimi_delta_attention_triton", aliases: ["kimi_delta_attention", "KDA triton", "KDA"] },
];

const OPERATOR_OWNER_RULES = {
  chunk_fwd_o: [{ owner: "吴雨舒" }],
  chunk_gated_delta_rule_fwd_h: [{ owner: "方梓阳" }],
  recompute_wu_fwd: [{ until: "2026-06-30", owner: "方梓阳" }, { owner: "周云飞" }],
  chunk_bwd_dv_local: [{ until: "2026-06-18", owner: "陈琳鑫" }, { owner: "叶倩雯" }],
  chunk_bwd_dqkwg: [{ until: "2026-06-30", owner: "黄浚哲" }, { owner: "李佳敏" }],
  chunk_gated_delta_rule_bwd_dhu: [{ owner: "方梓阳" }],
  prepare_wy_repr_bwd_da: [{ owner: "杨子奇" }],
  prepare_wy_repr_bwd_full: [{ until: "2026-06-30", owner: "张硕累" }, { owner: "周云飞" }],
};

const STATUS_OPTIONS = [["todo", "todo"], ["doing", "doing"], ["blocked", "Pending"], ["delayed", "delay"], ["done", "done"]];
const PL_OPTIONS = ["陈琳鑫", "赵臣臣", "唐超", "马越", "黄俊健", "龚翔宇", "周亭亭", "孙伟伟", "陈龙"];
const DEFAULT_PL = PL_OPTIONS[0];
const AUDIT_TASK_FIELDS = ["title", "operator_ids", "owner", "risk", "priority", "status", "group_id", "special_id", "start_date", "end_date", "recommit_date", "done_date", "pr_required", "pr_link", "test_report", "notes"];
const AUDIT_FIELD_LABELS = {
  title: "事项",
  operator_ids: "算子",
  owner: "责任人",
  risk: "风险",
  priority: "优先级",
  status: "状态",
  group_id: "分组",
  special_id: "专项",
  start_date: "开始日期",
  end_date: "结束日期",
  recommit_date: "延期承诺",
  done_date: "完成日期",
  pr_required: "PR 要求",
  pr_link: "PR 链接",
  test_report: "转测报告",
  notes: "备注",
  segments: "甘特分段",
};
const TABLE_SORT_LABELS = {
  risk: "风险",
  priority: "优先级",
  title: "事项",
  operator_ids: "算子",
  owner: "责任人",
  owner_pl: "责任人分组",
  group_id: "转测迭代计划排期",
  special_id: "专项",
  date: "距 DDL 工作日",
  recommit_date: "延期承诺",
  done_date: "完成日期",
  pr_required: "PR 要求",
  pr_link: "PR 链接",
  test_report: "转测报告",
  status: "状态",
};
const TABLE_SORT_DEFAULT_DIRECTION = {
  risk: "desc",
  priority: "asc",
  date: "asc",
  status: "asc",
};
const RISK_SORT_WEIGHT = { 高: 3, 中: 2, 低: 1 };
const PRIORITY_SORT_WEIGHT = { P0: 0, P1: 1, P2: 2 };
const STATUS_SORT_WEIGHT = { delayed: 0, blocked: 1, todo: 2, doing: 3, done: 4 };
const SCHEDULE_CHANGE_FIELDS = new Set(["start_date", "end_date", "segments"]);

const state = {
  data: null,
  prCatalog: { generatedAt: "", sourceRepo: "", total: 0, items: [] },
  audit: [],
  token: sessionStorage.getItem(WORKER_API_BASE ? "flashWorkerAuthToken" : "flashPagesToken") || "",
  authUser: null,
  serverVersion: "",
  pendingRemoteVersion: "",
  realtimeTimer: null,
  loading: false,
  dirtyTaskIds: new Set(),
  dirtyPersonIds: new Set(),
  taskBaselines: new Map(),
  personBaselines: new Map(),
  axis: { start: "", end: "", total: 1 },
  baseTimeline: { start: "", end: "", total: 1 },
  view: { start: "", end: "", total: 1 },
  timeline: { start: "", end: "", total: 1 },
  activePlanView: "timeline",
  filters: { q: "", risk: "", priority: "", owner: [], pl: "", operator_id: "", group_id: "", special_id: "", status: "" },
  sort: { field: "risk", direction: "desc" },
  ownerFilterOpen: false,
  ownerFilterQuery: "",
  ownerFilterDraft: [],
};

const $ = (selector) => document.querySelector(selector);

async function load() {
  if (state.loading) return;
  state.loading = true;
  const stamp = Date.now();
  try {
    if (WORKER_API_BASE) {
      const [data, audit, prCatalog, me] = await Promise.all([
        workerGet("/api/export"),
        workerGet("/api/audit?limit=10").catch(() => []),
        workerGet("/api/pr-catalog").catch(() => ({ generatedAt: "", sourceRepo: "", total: 0, items: [] })),
        state.token ? workerGet("/api/me").catch(() => null) : Promise.resolve(null),
      ]);
      if (state.token && !me) {
        state.token = "";
        state.authUser = null;
        sessionStorage.removeItem("flashWorkerAuthToken");
      } else if (me?.user) {
        state.authUser = me.user;
      }
      state.data = data;
      state.serverVersion = data.version || data.generatedAt || "";
      state.pendingRemoteVersion = "";
      state.dirtyTaskIds.clear();
      state.dirtyPersonIds.clear();
      state.taskBaselines.clear();
      state.personBaselines.clear();
      state.audit = audit;
      state.prCatalog = prCatalog || { generatedAt: "", sourceRepo: "", total: 0, items: [] };
      ensureOperatorCatalog();
      ensurePeopleCatalog();
      syncAllTaskDeliveryRules();
      render();
      return;
    }
    const [dataRes, auditRes, prCatalogRes] = await Promise.all([
      fetch(`./project-state.json?v=${stamp}`),
      fetch(`./audit-log.jsonl?v=${stamp}`).catch(() => null),
      fetch(`./pr-catalog.json?v=${stamp}`).catch(() => null),
    ]);
    if (!dataRes.ok) throw new Error("未读取到 project-state.json");
    state.data = await dataRes.json();
    state.audit = auditRes && auditRes.ok ? parseAudit(await auditRes.text()) : [];
    state.prCatalog = prCatalogRes && prCatalogRes.ok ? await prCatalogRes.json() : { generatedAt: "", sourceRepo: "", total: 0, items: [] };
    state.dirtyTaskIds.clear();
    state.dirtyPersonIds.clear();
    state.taskBaselines.clear();
    state.personBaselines.clear();
    ensureOperatorCatalog();
    ensurePeopleCatalog();
    syncAllTaskDeliveryRules();
    render();
  } finally {
    state.loading = false;
  }
}

function parseAudit(text) {
  return text.trim().split(/\n+/).filter(Boolean).map((line) => {
    try { return JSON.parse(line); } catch { return null; }
  }).filter(Boolean);
}

function taskSnapshot(task) {
  const snapshot = {};
  AUDIT_TASK_FIELDS.forEach((field) => {
    snapshot[field] = normalizeAuditValue(task?.[field]);
  });
  snapshot.segments = JSON.stringify((task?.segments || []).map((segment) => ({
    start_date: normalizeAuditValue(segment.start_date),
    end_date: normalizeAuditValue(segment.end_date),
    reason: normalizeAuditValue(segment.reason),
  })));
  return snapshot;
}

function normalizeAuditValue(value) {
  return value == null ? "" : String(value);
}

function rememberTaskBaseline(taskId) {
  if (!taskId || state.taskBaselines.has(taskId)) return;
  const task = state.data.tasks.find((item) => item.id === taskId);
  if (task) state.taskBaselines.set(taskId, taskSnapshot(task));
}

function taskAuditChange(task, beforeSnapshot = null) {
  if (!task) return null;
  const before = beforeSnapshot || state.taskBaselines.get(task.id) || taskSnapshot(task);
  const after = taskSnapshot(task);
  const changes = {};
  [...AUDIT_TASK_FIELDS, "segments"].forEach((field) => {
    if (before[field] !== after[field]) changes[field] = { from: before[field] || "", to: after[field] || "" };
  });
  return Object.keys(changes).length ? { id: task.id, title: task.title || task.id, changes } : null;
}

function auditDetailHtml(item) {
  const changes = item.detail?.changes || [];
  if (changes.length) {
    return `
      <details class="audit-detail">
        <summary>查看 ${changes.length} 项字段变更</summary>
        <div class="audit-change-list">${changes.map(auditTaskChangeHtml).join("")}</div>
      </details>
    `;
  }
  const ids = item.detail?.ids || [];
  if (ids.length) {
    return `
      <details class="audit-detail">
        <summary>查看记录到的 ${ids.length} 个任务 ID</summary>
        <div class="audit-id-list">${ids.map((id) => `<code>${escapeHtml(id)}</code>`).join("")}</div>
      </details>
    `;
  }
  return "";
}

function auditTaskChangeHtml(change) {
  const rows = Object.entries(change.changes || {}).map(([field, diff]) => `
    <li>
      <span>${escapeHtml(AUDIT_FIELD_LABELS[field] || field)}</span>
      <em>${escapeHtml(formatAuditValue(field, diff.from))}</em>
      <strong>→</strong>
      <em>${escapeHtml(formatAuditValue(field, diff.to))}</em>
    </li>
  `).join("");
  return `
    <section class="audit-change">
      <h3>${escapeHtml(change.title || change.id)}</h3>
      <ul>${rows}</ul>
    </section>
  `;
}

function formatAuditValue(field, value) {
  const text = normalizeAuditValue(value);
  if (!text) return "空";
  if (field === "status") return statusLabel(text);
  if (field === "pr_required") return taskRequiresPr({ pr_required: text }) ? "需 PR" : "仅报告";
  if (field === "operator_ids") return operatorLabelsForIds(text).join(" / ") || "空";
  if (field === "group_id") return groupTitle(text) || text;
  if (field === "special_id") return specialTitle(text) || text;
  if (field === "segments") return formatAuditSegments(text);
  return text;
}

function formatAuditSegments(value) {
  try {
    const segments = JSON.parse(value);
    if (!segments.length) return "空";
    return segments.map((segment) => {
      const range = `${segment.start_date || "?"} ~ ${segment.end_date || "?"}`;
      return segment.reason ? `${range}（${segment.reason}）` : range;
    }).join("；");
  } catch {
    return value || "空";
  }
}

function filteredTasks() {
  const tasks = visibleTasksForCurrentUser();
  return tasks.filter((task) => {
    const q = state.filters.q.toLowerCase();
    const ownerNames = taskOwnerNames(task);
    const ownerFilters = ownerFilterValues();
    const plNames = taskPlNames(task);
    const operatorNames = taskOperators(task).map((operator) => operator.label).join(" ");
    return (!q || [task.title, task.owner, task.scope, ownerNames.join(" "), plNames.join(" "), operatorNames].some((value) => String(value || "").toLowerCase().includes(q)))
      && (!state.filters.risk || task.risk === state.filters.risk)
      && (!state.filters.priority || task.priority === state.filters.priority)
      && (!ownerFilters.length || ownerFilters.some((name) => ownerNames.includes(name)))
      && (!state.filters.pl || plNames.includes(state.filters.pl))
      && (!state.filters.operator_id || (state.filters.operator_id === "__none__"
        ? !taskOperators(task).length
        : taskOperators(task).some((operator) => operator.id === state.filters.operator_id)))
      && (!state.filters.group_id || task.group_id === state.filters.group_id)
      && (!state.filters.special_id || (state.filters.special_id === "__none__" ? !task.special_id : task.special_id === state.filters.special_id))
      && (!state.filters.status || task.status === state.filters.status);
  });
}

function sortTasksForTable(tasks) {
  const field = state.sort?.field || "risk";
  const direction = state.sort?.direction === "asc" ? "asc" : "desc";
  const factor = direction === "asc" ? 1 : -1;
  return [...tasks].sort((a, b) => {
    const primary = compareTaskByField(a, b, field);
    if (primary) return primary * factor;
    const fallback = compareTaskByField(a, b, "risk") * -1
      || compareTaskByField(a, b, "priority")
      || compareTaskByField(a, b, "date")
      || displayTaskTitle(a).localeCompare(displayTaskTitle(b), "zh-CN")
      || String(a.id || "").localeCompare(String(b.id || ""));
    return fallback;
  });
}

function compareTaskByField(a, b, field) {
  if (field === "risk") return numberCompare(RISK_SORT_WEIGHT[a.risk] || 0, RISK_SORT_WEIGHT[b.risk] || 0);
  if (field === "priority") return numberCompare(PRIORITY_SORT_WEIGHT[a.priority] ?? 99, PRIORITY_SORT_WEIGHT[b.priority] ?? 99);
  if (field === "status") return numberCompare(STATUS_SORT_WEIGHT[a.status] ?? 99, STATUS_SORT_WEIGHT[b.status] ?? 99);
  if (field === "date") {
    return String(taskSortDdl(a)).localeCompare(String(taskSortDdl(b)))
      || String(taskSortStart(a)).localeCompare(String(taskSortStart(b)));
  }
  if (field === "recommit_date" || field === "done_date") return String(a[field] || "").localeCompare(String(b[field] || ""));
  if (field === "pr_required") return numberCompare(normalizeBooleanFlag(b.pr_required, true), normalizeBooleanFlag(a.pr_required, true));
  if (field === "group_id") return groupTitle(a.group_id).localeCompare(groupTitle(b.group_id), "zh-CN");
  if (field === "special_id") return specialTitle(a.special_id).localeCompare(specialTitle(b.special_id), "zh-CN");
  if (field === "operator_ids") return taskOperatorLabelText(a).localeCompare(taskOperatorLabelText(b), "zh-CN");
  if (field === "owner") return taskOwnerNames(a).join("、").localeCompare(taskOwnerNames(b).join("、"), "zh-CN");
  if (field === "owner_pl") return taskPlNames(a).join("、").localeCompare(taskPlNames(b).join("、"), "zh-CN");
  if (field === "title") return displayTaskTitle(a).localeCompare(displayTaskTitle(b), "zh-CN");
  if (field === "pr_link" || field === "test_report") return String(a[field] || "").localeCompare(String(b[field] || ""), "zh-CN");
  return 0;
}

function taskSortDdl(task) {
  return task.end_date || taskSegments(task).map((segment) => segment.end_date).filter(Boolean).sort().at(-1) || "";
}

function taskSortStart(task) {
  return task.start_date || taskSegments(task).map((segment) => segment.start_date).filter(Boolean).sort()[0] || "";
}

function numberCompare(a, b) {
  return Number(a) - Number(b);
}

function normalizeBooleanFlag(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback ? 1 : 0;
  if (value === true || value === 1) return 1;
  if (value === false || value === 0) return 0;
  const text = String(value).trim().toLowerCase();
  if (["1", "true", "yes", "y", "on"].includes(text)) return 1;
  if (["0", "false", "no", "n", "off"].includes(text)) return 0;
  return fallback ? 1 : 0;
}

function ownerFilterValues() {
  const owner = state.filters.owner;
  if (Array.isArray(owner)) return owner.filter(Boolean);
  return owner ? [owner] : [];
}

function visibleTasksForCurrentUser() {
  const tasks = state.data?.tasks || [];
  if (!isDeveloperEditMode()) return tasks;
  const relatedOperatorIds = developerOwnedOperatorIdsInCurrentView();
  return tasks.filter((task) => canEditTask(task) || taskOperators(task).some((operator) => relatedOperatorIds.has(operator.id)));
}

function isDeveloperEditMode() {
  return Boolean(WORKER_API_BASE && state.token && state.authUser?.role === "developer");
}

function isAdminEditMode() {
  return Boolean(state.token && (!WORKER_API_BASE || state.authUser?.role === "admin"));
}

function currentDeveloperOwnerName() {
  if (!state.authUser) return "";
  return normalizeOwnerName(state.authUser.ownerName || state.authUser.displayName || state.authUser.username);
}

function canEditTask(task) {
  if (!isDeveloperEditMode()) return Boolean(state.token);
  const ownerName = currentDeveloperOwnerName();
  return Boolean(ownerName && taskOwnerNames(task).includes(ownerName));
}

function canScheduleTask(task) {
  return Boolean(task && isAdminEditMode());
}

function developerOwnedOperatorIdsInCurrentView() {
  const ownerName = currentDeveloperOwnerName();
  if (!ownerName || !isYmd(state.view.start) || !isYmd(state.view.end)) return new Set();
  const ownedTasks = (state.data?.tasks || []).filter((task) => {
    return taskOwnerNames(task).includes(ownerName) && taskIntersectsView(task);
  });
  return new Set(ownedTasks.flatMap((task) => taskOperators(task).map((operator) => operator.id)));
}

function render(options = {}) {
  const includeTableFilters = options.includeTableFilters !== false;
  const all = state.data.tasks || [];
  ensureDefaultTimelineView();
  state.baseTimeline = computeTimelineRange();
  ensureTimelineView();
  const scoped = visibleTasksForCurrentUser();
  const filtered = filteredTasks();
  const tasks = filtered.filter(taskIntersectsView);
  const high = tasks.filter((task) => task.risk === "高").length;
  const medium = tasks.filter((task) => task.risk === "中").length;
  const done = tasks.filter((task) => task.status === "done").length;
  const scopeText = scoped.length === all.length
    ? `${tasks.length}/${filtered.length}/${all.length}`
    : `${tasks.length}/${filtered.length}/${scoped.length}（权限范围，全量 ${all.length}）`;
  $("#meta").textContent = `仓库数据更新时间：${state.data.generatedAt || "未知"} · 窗口 ${state.view.start} ~ ${state.view.end} · 当前显示 ${scopeText} 项`;
  $("#summary").innerHTML = [
    ["总任务", tasks.length],
    ["高风险", high],
    ["中风险", medium],
    ["已完成", done],
  ].map(([label, value]) => `<div class="metric"><small>${label}</small><strong>${value}</strong></div>`).join("");
  const workerMode = Boolean(WORKER_API_BASE);
  document.body.classList.toggle("editing", Boolean(state.token));
  document.body.classList.toggle("developer-editing", isDeveloperEditMode());
  document.body.classList.toggle("admin-editing", isAdminEditMode());
  document.body.classList.toggle("worker-mode", workerMode);
  $("#token").value = state.token ? "********" : "";
  $(".token-box").classList.toggle("hidden", workerMode);
  $("#workerLogin").classList.toggle("hidden", !workerMode || Boolean(state.token));
  $("#logout").classList.toggle("hidden", !state.token);
  $("#changePassword").classList.toggle("hidden", !workerMode || !state.token || !state.authUser || state.authUser.id === "admin-token");
  $("#editMode").classList.toggle("hidden", Boolean(state.token) || workerMode);
  updateEditStatus();
  renderPlanTabs();
  renderTimeAxis();
  const timelineTasks = tasks.filter(taskVisibleOnTimeline);
  renderGantt(timelineTasks);
  renderPeopleView(timelineTasks);
  renderOperatorView(timelineTasks);
  if (includeTableFilters) renderTableFilters();
  else updateOwnerFilterSummary();
  renderTableSortHeaders();
  renderRows(sortTasksForTable(tasks));
  renderAdmin();
  renderAudit();
}

function renderPlanTabs() {
  document.querySelectorAll("[data-plan-tab]").forEach((button) => {
    const active = button.dataset.planTab === state.activePlanView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.onclick = () => {
      state.activePlanView = button.dataset.planTab;
      renderPlanTabs();
    };
  });
  document.querySelectorAll("[data-plan-view]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.planView === state.activePlanView);
  });
}

function renderTableFilters() {
  const scopedTasks = visibleTasksForCurrentUser();
  const visibleGroupIds = new Set(scopedTasks.map((task) => task.group_id).filter(Boolean));
  const visibleSpecialIds = new Set(scopedTasks.map((task) => task.special_id).filter(Boolean));
  const visibleOperatorIds = new Set(scopedTasks.flatMap((task) => taskOperators(task).map((operator) => operator.id)));
  const groups = isDeveloperEditMode() ? state.data.groups.filter((group) => visibleGroupIds.has(group.id)) : state.data.groups;
  const specials = isDeveloperEditMode() ? state.data.specials.filter((special) => visibleSpecialIds.has(special.id)) : state.data.specials;
  const operators = isDeveloperEditMode() ? operatorCatalog().filter((operator) => visibleOperatorIds.has(operator.id)) : operatorCatalog();
  const columns = [
    tableFilterSelect("risk", [["", "全部"], ["高", "高"], ["中", "中"], ["低", "低"]]),
    tableFilterSelect("priority", [["", "全部"], ["P0", "P0"], ["P1", "P1"], ["P2", "P2"]]),
    `<th><input data-table-filter="q" type="search" placeholder="筛事项" value="${escapeAttr(state.filters.q)}"></th>`,
    tableFilterSelect("operator_id", [["", "全部"], ["__none__", "未关联算子"], ...operators.map((operator) => [operator.id, operator.label])], "col-operator"),
    ownerFilterDropdown(),
    tableFilterSelect("pl", [["", "全部"], ...plFilterOptions().map((pl) => [pl, pl])], "col-pl"),
    tableFilterSelect("group_id", [["", "全部"], ...groups.map((group) => [group.id, group.title])], "col-group"),
    tableFilterSelect("special_id", [["", "全部"], ["__none__", "普通事项"], ...specials.map((special) => [special.id, special.title])], "col-special"),
    `<th class="col-date"></th>`,
    `<th class="col-commit"></th>`,
    `<th class="col-done-date"></th>`,
    `<th class="col-pr-required"></th>`,
    `<th class="col-pr"></th>`,
    `<th class="col-report"></th>`,
    tableFilterSelect("status", [["", "全部"], ...STATUS_OPTIONS]),
    `<th class="edit-only"><button type="button" data-clear-filters>清空</button></th>`,
  ];
  $("#tableFilters").innerHTML = columns.join("");
  document.querySelectorAll("[data-table-filter]").forEach((control) => {
    control.addEventListener("input", updateTableFilter);
    control.addEventListener("change", updateTableFilter);
  });
  document.querySelector("[data-owner-filter]")?.addEventListener("click", (event) => event.stopPropagation());
  document.querySelector("[data-owner-filter-toggle]")?.addEventListener("click", toggleOwnerFilter);
  document.querySelector("[data-owner-filter-search]")?.addEventListener("input", updateOwnerFilterQuery);
  document.querySelectorAll("[data-owner-filter-value]").forEach((control) => {
    control.addEventListener("click", updateOwnerFilter);
    control.addEventListener("change", updateOwnerFilter);
  });
  document.querySelector("[data-owner-filter-select-all]")?.addEventListener("click", selectAllOwnerFilter);
  document.querySelector("[data-owner-filter-clear]")?.addEventListener("click", clearOwnerFilter);
  document.querySelector("[data-owner-filter-apply]")?.addEventListener("click", applyOwnerFilter);
  document.querySelector("[data-clear-filters]")?.addEventListener("click", clearFilters);
  positionOwnerFilterMenu();
}

function renderTableSortHeaders() {
  document.querySelectorAll("[data-sort-field]").forEach((button) => {
    const field = button.dataset.sortField;
    const active = state.sort.field === field;
    const direction = active ? state.sort.direction : "";
    button.classList.toggle("active", active);
    button.setAttribute("aria-sort", active ? (direction === "asc" ? "ascending" : "descending") : "none");
    button.title = `按${TABLE_SORT_LABELS[field] || field}排序`;
    const indicator = button.querySelector("[data-sort-indicator]");
    if (indicator) indicator.textContent = active ? (direction === "asc" ? "↑" : "↓") : "↕";
    button.onclick = () => updateTableSort(field);
  });
}

function updateTableSort(field) {
  if (!TABLE_SORT_LABELS[field]) return;
  if (state.sort.field === field) {
    state.sort.direction = state.sort.direction === "asc" ? "desc" : "asc";
  } else {
    state.sort = { field, direction: TABLE_SORT_DEFAULT_DIRECTION[field] || "asc" };
  }
  render({ includeTableFilters: false });
}

function tableFilterSelect(field, options, className = "") {
  const seen = new Set();
  const normalized = options.filter(([id]) => {
    const key = String(id);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const klass = className ? ` class="${escapeAttr(className)}"` : "";
  return `<th${klass}><select data-table-filter="${field}">${normalized.map(([id, label]) => `<option value="${escapeAttr(id)}" ${state.filters[field] === id ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select></th>`;
}

function ownerFilterDropdown() {
  const selected = state.ownerFilterOpen ? state.ownerFilterDraft : ownerFilterValues();
  const selectedSet = new Set(selected);
  const label = ownerFilterLabel();
  return `
    <th>
      <div class="check-filter ${state.ownerFilterOpen ? "open" : ""}" data-owner-filter>
        <button type="button" class="check-filter-trigger" data-owner-filter-toggle aria-expanded="${state.ownerFilterOpen ? "true" : "false"}">
          <span>${escapeHtml(label)}</span>
        </button>
        <div class="check-filter-menu">
          <input class="check-filter-search" data-owner-filter-search type="search" placeholder="搜索责任人" value="${escapeAttr(state.ownerFilterQuery)}">
          <div class="check-filter-options">
            ${ownerFilterOptionsHtml(selectedSet)}
          </div>
          <div class="check-filter-actions">
            <button type="button" class="check-filter-select-all" data-owner-filter-select-all>全选</button>
            <button type="button" class="check-filter-clear" data-owner-filter-clear>清空</button>
            <button type="button" class="check-filter-apply" data-owner-filter-apply>应用</button>
          </div>
        </div>
      </div>
    </th>
  `;
}

function ownerFilterOptionsHtml(selectedSet = new Set(state.ownerFilterDraft)) {
  const query = state.ownerFilterQuery.trim().toLowerCase();
  const seen = new Set();
  const normalized = ownerFilterOptions()
    .filter(([id]) => !query || id.toLowerCase().includes(query))
    .filter(([id]) => {
      const key = String(id);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  return normalized.length ? normalized.map(([id, optionLabel]) => `
    <label class="check-option">
      <input type="checkbox" data-owner-filter-value value="${escapeAttr(id)}" ${selectedSet.has(id) ? "checked" : ""}>
      <span>${escapeHtml(optionLabel)}</span>
    </label>
  `).join("") : `<div class="check-filter-empty">无匹配责任人</div>`;
}

function refreshOwnerFilterOptions() {
  const container = document.querySelector(".check-filter-options");
  if (!container) return renderTableFilters();
  container.innerHTML = ownerFilterOptionsHtml(new Set(state.ownerFilterDraft));
  container.querySelectorAll("[data-owner-filter-value]").forEach((control) => {
    control.addEventListener("click", updateOwnerFilter);
    control.addEventListener("change", updateOwnerFilter);
  });
}

function ownerFilterLabel() {
  const selected = ownerFilterValues();
  return selected.length ? `已选 ${selected.length} 人` : "全部";
}

function updateOwnerFilterSummary() {
  const label = document.querySelector("[data-owner-filter-toggle] span");
  if (label) label.textContent = ownerFilterLabel();
  positionOwnerFilterMenu();
}

function positionOwnerFilterMenu() {
  if (!state.ownerFilterOpen) return;
  const trigger = document.querySelector("[data-owner-filter-toggle]");
  const menu = document.querySelector(".check-filter-menu");
  if (!trigger || !menu) return;
  const rect = trigger.getBoundingClientRect();
  const menuWidth = menu.offsetWidth || 220;
  const menuHeight = menu.offsetHeight || 320;
  const left = clamp(rect.left, 8, Math.max(8, window.innerWidth - menuWidth - 8));
  let top = rect.bottom + 4;
  if (top + menuHeight > window.innerHeight - 8) {
    top = Math.max(8, rect.top - menuHeight - 4);
  }
  menu.style.setProperty("--filter-menu-left", `${left}px`);
  menu.style.setProperty("--filter-menu-top", `${top}px`);
}

function uniqueTaskValues(field) {
  return [...new Set((state.data.tasks || []).map((task) => task[field]).filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b), "zh-CN"));
}

function ownerFilterOptions() {
  const tasks = visibleTasksForCurrentUser();
  const visiblePeople = isDeveloperEditMode()
    ? (state.data.people || []).filter((person) => tasks.some((task) => taskOwnerNames(task).includes(person.name)))
    : (state.data.people || []);
  return uniqueStrings([
    ...visiblePeople.map((person) => person.name),
    ...tasks.flatMap((task) => taskOwnerNames(task)),
    ...(isDeveloperEditMode() ? [] : operatorOwnerRuleNames()),
  ])
    .filter(isSelectableOwnerName)
    .sort((a, b) => a.localeCompare(b, "zh-CN"))
    .map((name) => [name, name]);
}

function plFilterOptions() {
  return uniqueStrings([
    ...PL_OPTIONS,
    ...(state.data.people || []).map((person) => normalizePl(person.pl)),
    ...(visibleTasksForCurrentUser() || []).flatMap(taskPlNames),
  ]).sort((a, b) => a.localeCompare(b, "zh-CN"));
}

function operatorOwnerRuleNames() {
  return uniqueStrings(operatorCatalog().flatMap((operator) => operator.owner_rules || []).map((rule) => rule.owner));
}

function isSelectableOwnerName(name) {
  const normalized = normalizeOwnerName(name);
  return Boolean(normalized && !/[、/,，;；&]/.test(normalized) && (!isPlaceholderOwner(normalized) || normalized === "待排人力"));
}

function updateTableFilter(event) {
  const field = event.target.dataset.tableFilter;
  const anchor = captureViewportAnchor(event.target);
  state.filters[field] = event.target.value.trim();
  syncToolbarFilters();
  render({ includeTableFilters: false });
  restoreViewportAnchor(anchor);
}

function captureViewportAnchor(element) {
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  return { element, top: rect.top };
}

function restoreViewportAnchor(anchor) {
  if (!anchor?.element?.isConnected) return;
  const rect = anchor.element.getBoundingClientRect();
  const delta = rect.top - anchor.top;
  if (Math.abs(delta) > 0.5) window.scrollBy(0, delta);
}

function toggleOwnerFilter(event) {
  event.stopPropagation();
  state.ownerFilterOpen = !state.ownerFilterOpen;
  if (state.ownerFilterOpen) state.ownerFilterDraft = ownerFilterValues();
  renderTableFilters();
}

function updateOwnerFilterQuery(event) {
  state.ownerFilterQuery = event.target.value;
  state.ownerFilterOpen = true;
  refreshOwnerFilterOptions();
  positionOwnerFilterMenu();
}

function updateOwnerFilter(event) {
  const selected = new Set(state.ownerFilterDraft);
  if (event.target.checked) selected.add(event.target.value);
  else selected.delete(event.target.value);
  state.ownerFilterDraft = [...selected].sort((a, b) => a.localeCompare(b, "zh-CN"));
  state.ownerFilterOpen = true;
}

function selectAllOwnerFilter(event) {
  event.stopPropagation();
  const query = state.ownerFilterQuery.trim().toLowerCase();
  const visibleOwners = ownerFilterOptions()
    .map(([id]) => id)
    .filter((id) => !query || id.toLowerCase().includes(query));
  const selected = new Set(state.ownerFilterDraft);
  visibleOwners.forEach((owner) => selected.add(owner));
  state.ownerFilterDraft = [...selected].sort((a, b) => a.localeCompare(b, "zh-CN"));
  document.querySelectorAll("[data-owner-filter-value]").forEach((control) => {
    control.checked = state.ownerFilterDraft.includes(control.value);
  });
}

function clearOwnerFilter(event) {
  event.stopPropagation();
  state.ownerFilterDraft = [];
  state.filters.owner = [];
  state.ownerFilterQuery = "";
  state.ownerFilterOpen = false;
  syncToolbarFilters();
  render();
}

function applyOwnerFilter(event) {
  event.stopPropagation();
  state.filters.owner = [...state.ownerFilterDraft].sort((a, b) => a.localeCompare(b, "zh-CN"));
  state.ownerFilterOpen = false;
  state.ownerFilterQuery = "";
  syncToolbarFilters();
  render();
}

function clearFilters() {
  Object.keys(state.filters).forEach((key) => { state.filters[key] = key === "owner" ? [] : ""; });
  state.ownerFilterOpen = false;
  state.ownerFilterQuery = "";
  state.ownerFilterDraft = [];
  syncToolbarFilters();
  render();
}

function syncToolbarFilters() {
  ["q", "risk", "priority", "status"].forEach((id) => {
    const control = $(`#${id}`);
    if (control) control.value = state.filters[id] || "";
  });
}

function computeTimelineRange() {
  return timelineRangeFromDates(state.axis.start, state.axis.end);
}

function computeDataTimelineRange() {
  const dates = [];
  (state.data?.tasks || []).forEach((task) => {
    if (isYmd(task.recommit_date)) dates.push(task.recommit_date);
    if (isYmd(task.done_date)) dates.push(task.done_date);
    taskRenderSegments(task).forEach((segment) => {
      if (segment.start_date) dates.push(segment.start_date);
      if (segment.end_date) dates.push(segment.end_date);
    });
  });
  (state.data?.groups || []).forEach((group) => {
    [group.start_date, group.end_date, group.due_date].forEach((date) => {
      if (date) dates.push(date);
    });
  });
  const parsed = dates.map(Date.parse).filter(Number.isFinite);
  if (!parsed.length) {
    const today = toDay(Date.now());
    return { start: today, end: today, total: 1 };
  }
  const start = toDay(Math.min(...parsed));
  const end = toDay(Math.max(...parsed));
  return timelineRangeFromDates(start, end);
}

function ensureDefaultTimelineView() {
  if (!isYmd(state.axis.start) || !isYmd(state.axis.end)) {
    const start = todayBjYmd();
    const end = maxDate(start, latestPlannedCompletionDate() || start);
    state.axis = timelineRangeFromDates(start, end);
  }
  if (isYmd(state.view.start) && isYmd(state.view.end)) return;
  state.view = { ...state.axis };
}

function latestPlannedCompletionDate() {
  const taskDates = [];
  (state.data?.tasks || []).forEach((task) => {
    if (isYmd(task.end_date)) taskDates.push(task.end_date);
    if (isYmd(task.recommit_date)) taskDates.push(task.recommit_date);
    if (isYmd(task.done_date)) taskDates.push(task.done_date);
    taskSegments(task).forEach((segment) => {
      if (isYmd(segment.end_date)) taskDates.push(segment.end_date);
    });
  });
  if (taskDates.length) return taskDates.sort().at(-1);
  const groupDates = [];
  (state.data?.groups || []).forEach((group) => {
    [group.end_date, group.due_date, group.start_date].forEach((date) => {
      if (isYmd(date)) groupDates.push(date);
    });
  });
  return groupDates.sort().at(-1) || "";
}

function timelineRangeFromDates(start, end) {
  const safeStart = minDate(start, end);
  const safeEnd = maxDate(start, end);
  return { start: safeStart, end: safeEnd, total: Math.max(1, daysBetween(safeStart, safeEnd) + 1) };
}

function ensureTimelineView() {
  const full = state.baseTimeline;
  if (!state.view.start || !state.view.end) {
    setTimelineView(0, full.total - 1);
    return;
  }
  const rawStart = daysBetween(full.start, state.view.start);
  const rawEnd = daysBetween(full.start, state.view.end);
  const startOffset = Number.isFinite(rawStart) ? clamp(rawStart, 0, full.total - 1) : 0;
  const endOffset = Number.isFinite(rawEnd) ? clamp(rawEnd, startOffset, full.total - 1) : full.total - 1;
  setTimelineView(startOffset, endOffset);
}

function setTimelineView(startOffset, endOffset) {
  const full = state.baseTimeline;
  const safeStart = clamp(Math.round(startOffset), 0, Math.max(0, full.total - 1));
  const safeEnd = clamp(Math.round(endOffset), safeStart, Math.max(safeStart, full.total - 1));
  const start = addDays(full.start, safeStart);
  const end = addDays(full.start, safeEnd);
  state.view = { start, end, total: safeEnd - safeStart + 1 };
  state.timeline = { ...state.view };
}

function taskSegments(task) {
  return task.segments?.length ? task.segments : [{ start_date: task.start_date, end_date: task.end_date, reason: task.notes || "", position: 0 }];
}

function taskRenderSegments(task) {
  const segments = taskSegments(task).map((segment, index) => ({ ...segment, source_index: index, auto_extended: false }));
  if (!segments.length || evaluateTaskDelivery(task).status !== "delayed") return segments;
  const today = todayBjYmd();
  let latestIndex = 0;
  segments.forEach((segment, index) => {
    if ((segment.end_date || "") > (segments[latestIndex].end_date || "")) latestIndex = index;
  });
  const latestEnd = segments[latestIndex].end_date;
  if (latestEnd && latestEnd < today) {
    const delayStart = addDays(latestEnd, 1);
    segments.push({
      id: `delay-${segments[latestIndex].id || latestIndex}`,
      start_date: delayStart,
      end_date: today,
      reason: "自动延长",
      position: segments.length,
      source_index: latestIndex,
      auto_extended: true,
    });
  }
  return segments;
}

function taskEarlySavedSegments(task) {
  if (evaluateTaskDelivery(task).status !== "done" || !isYmd(task.done_date)) return [];
  return taskSegments(task).flatMap((segment, index) => {
    if (!isYmd(segment.end_date) || task.done_date >= segment.end_date) return [];
    const start = maxDate(addDays(task.done_date, 1), segment.start_date);
    const end = segment.end_date;
    return start <= end ? [{
      id: `saved-${segment.id || index}`,
      start_date: start,
      end_date: end,
      source_index: index,
      saved: true,
    }] : [];
  });
}

function taskRecommitMarkerHtml(task, start, total) {
  if (evaluateTaskDelivery(task).status !== "delayed" || !isYmd(task.recommit_date)) return "";
  if (task.recommit_date < start || task.recommit_date > addDays(start, total - 1)) return "";
  const left = slotCenterPct(daysBetween(start, task.recommit_date), total);
  return `<span class="recommit-marker" style="--x:${left}%" title="${escapeAttr(displayTaskTitle(task))}：延期承诺 ${escapeAttr(task.recommit_date)}"><i></i><small>${escapeHtml(formatMonthDay(task.recommit_date))}</small></span>`;
}

function maxDelayedRenderEndDate() {
  const delayedEnds = (state.data?.tasks || []).flatMap((task) => taskRenderSegments(task)
    .filter((segment) => segment.auto_extended)
    .map((segment) => segment.end_date));
  return delayedEnds.sort().at(-1) || "";
}

function taskRenderStart(task) {
  return taskRenderSegments(task).map((segment) => segment.start_date).filter(Boolean).sort()[0] || task.start_date;
}

function taskRenderEnd(task) {
  return taskRenderSegments(task).map((segment) => segment.end_date).filter(Boolean).sort().at(-1) || task.end_date;
}

function taskIntersectsView(task) {
  return taskRenderSegments(task).some((segment) => datesOverlap(segment.start_date, segment.end_date, state.view.start, state.view.end));
}

function datesOverlap(startA, endA, startB, endB) {
  return Boolean(startA && endA && startB && endB && startA <= endB && endA >= startB);
}

function renderTimeAxis() {
  const full = state.baseTimeline;
  const overviewMilestones = timelineMilestones(full);
  const overviewTicks = timelineTicks(overviewMilestones, full);
  const detail = state.timeline;
  const detailMilestones = timelineMilestones(detail);
  const detailTicks = timelineTicks(detailMilestones, detail);
  $("#timeAxis").innerHTML = `
    <div class="timeline-head">
      <div>
        <strong>DDL 与时间窗口</strong>
        <span>开始/结束定义整条时间轴；绿色窗口为当前显示范围，初始铺满时间轴，可在轴内拖动或缩放</span>
      </div>
      <div class="timeline-actions">
        <label>开始 <input id="viewStartDate" type="date" value="${escapeAttr(state.axis.start)}"></label>
        <label>结束 <input id="viewEndDate" type="date" value="${escapeAttr(state.axis.end)}"></label>
        <button id="expandStart" type="button">前扩 7 天</button>
        <button id="expandEnd" type="button">后扩 7 天</button>
        <button id="resetTimeline" type="button">显示全量</button>
      </div>
    </div>
    <div id="timelineScale" class="timeline-scale overview-scale">
      <div class="axis-line"></div>
      ${overviewTicks.map((offset) => `
        <span class="axis-tick" style="left:${slotCenterPct(offset, full.total)}%">
          <i></i><em>${formatMonthDay(addDays(full.start, offset))}</em>
        </span>
      `).join("")}
      ${overviewMilestones.map((item) => `
        <span class="ddl-marker" style="left:${slotCenterPct(item.offset, full.total)}%" title="${escapeAttr(item.title)}：${escapeAttr(item.date)}">
          <i></i>
          <b>DDL</b>
          <em>${escapeHtml(formatMonthDay(item.date))}</em>
          <small>${escapeHtml(item.title)}</small>
        </span>
      `).join("")}
      <div class="timeline-window" data-axis-mode="move">
        <span class="timeline-window-edge" data-axis-mode="start"></span>
        <span class="timeline-window-label" data-axis-mode="move" id="viewText"></span>
        <span class="timeline-window-edge" data-axis-mode="end"></span>
      </div>
    </div>
    <div class="detail-axis-grid">
      <div class="detail-axis-label">当前窗口</div>
      <div class="detail-scale">
        <div class="axis-line"></div>
        ${detailTicks.map((offset) => `
          <span class="axis-tick" style="left:${slotCenterPct(offset, detail.total)}%">
            <i></i><em>${formatMonthDay(addDays(detail.start, offset))}</em>
          </span>
        `).join("")}
        ${detailMilestones.map((item) => `
          <span class="ddl-marker" style="left:${slotCenterPct(item.offset, detail.total)}%" title="${escapeAttr(item.title)}：${escapeAttr(item.date)}">
            <i></i>
            <b>DDL</b>
            <em>${escapeHtml(formatMonthDay(item.date))}</em>
          </span>
        `).join("")}
      </div>
    </div>
  `;
  paintTimelineWindow();
  attachTimelineDrag();
  $("#viewStartDate").addEventListener("change", applyTimelineDateInputs);
  $("#viewEndDate").addEventListener("change", applyTimelineDateInputs);
  $("#expandStart").addEventListener("click", () => setAbsoluteTimelineView(addDays(state.axis.start, -7), state.axis.end));
  $("#expandEnd").addEventListener("click", () => setAbsoluteTimelineView(state.axis.start, addDays(state.axis.end, 7)));
  $("#resetTimeline").addEventListener("click", () => {
    const dataRange = computeDataTimelineRange();
    state.axis = dataRange;
    state.baseTimeline = { ...dataRange };
    state.view = { ...dataRange };
    state.timeline = { ...dataRange };
    render();
  });
}

function applyTimelineDateInputs() {
  const start = $("#viewStartDate").value;
  const end = $("#viewEndDate").value;
  setAbsoluteTimelineView(start, end);
}

function setAbsoluteTimelineView(start, end) {
  if (!isYmd(start) || !isYmd(end)) {
    alert("请输入有效日期。");
    return;
  }
  const nextStart = minDate(start, end);
  const nextEnd = maxDate(start, end);
  state.axis = timelineRangeFromDates(nextStart, nextEnd);
  state.baseTimeline = { ...state.axis };
  state.view = { ...state.axis };
  state.timeline = { ...state.axis };
  render();
}

function timelineMilestones(range = state.baseTimeline) {
  return (state.data?.groups || []).map((group) => {
    const date = group.due_date || group.end_date || group.start_date;
    const offset = date ? daysBetween(range.start, date) : NaN;
    return { title: group.title, date, offset };
  }).filter((item) => item.date && Number.isFinite(item.offset) && item.offset >= 0 && item.offset < range.total)
    .sort((a, b) => a.offset - b.offset || a.title.localeCompare(b.title, "zh-CN"));
}

function timelineTicks(milestones, range = state.baseTimeline) {
  const step = range.total <= 18 ? 1 : Math.ceil(range.total / 16);
  const offsets = new Set([0, Math.max(0, range.total - 1), ...milestones.map((item) => item.offset)]);
  for (let offset = 0; offset < range.total; offset += step) offsets.add(offset);
  return [...offsets].filter((offset) => offset >= 0 && offset < range.total).sort((a, b) => a - b);
}

function attachTimelineDrag() {
  const scale = $("#timelineScale");
  if (!scale) return;
  scale.addEventListener("pointerdown", (event) => {
    const target = event.target.closest("[data-axis-mode]");
    const mode = target?.dataset.axisMode || "jump";
    const full = state.baseTimeline;
    const startOffset = daysBetween(full.start, state.view.start);
    const endOffset = daysBetween(full.start, state.view.end);
    const span = endOffset - startOffset + 1;
    const firstOffset = pointerToTimelineOffset(event, scale);
    const grabOffset = firstOffset - startOffset;
    event.preventDefault();

    if (mode === "jump") {
      const nextStart = clamp(firstOffset - Math.floor(span / 2), 0, Math.max(0, full.total - span));
      setTimelineView(nextStart, nextStart + span - 1);
      render();
      return;
    }

    scale.setPointerCapture(event.pointerId);
    scale.classList.add("dragging");
    const move = (moveEvent) => {
      const offset = pointerToTimelineOffset(moveEvent, scale);
      let nextStart = startOffset;
      let nextEnd = endOffset;
      if (mode === "move") {
        nextStart = clamp(offset - grabOffset, 0, Math.max(0, full.total - span));
        nextEnd = nextStart + span - 1;
      } else if (mode === "start") {
        nextStart = clamp(offset, 0, endOffset);
      } else {
        nextEnd = clamp(offset, startOffset, full.total - 1);
      }
      setTimelineView(nextStart, nextEnd);
      paintTimelineWindow();
    };
    const finish = () => {
      scale.removeEventListener("pointermove", move);
      scale.removeEventListener("pointerup", finish);
      scale.removeEventListener("pointercancel", finish);
      scale.classList.remove("dragging");
      render();
    };
    scale.addEventListener("pointermove", move);
    scale.addEventListener("pointerup", finish);
    scale.addEventListener("pointercancel", finish);
  });
}

function pointerToTimelineOffset(event, scale) {
  const rect = scale.getBoundingClientRect();
  const ratio = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
  return clamp(Math.floor(ratio * state.baseTimeline.total), 0, state.baseTimeline.total - 1);
}

function paintTimelineWindow() {
  const full = state.baseTimeline;
  const startOffset = daysBetween(full.start, state.view.start);
  const endOffset = daysBetween(full.start, state.view.end);
  const left = slotStartPct(startOffset, full.total);
  const width = Math.max(1.5, (endOffset - startOffset + 1) / Math.max(1, full.total) * 100);
  const windowEl = document.querySelector(".timeline-window");
  const label = document.querySelector(".timeline-window-label");
  if (windowEl) {
    windowEl.style.left = `${left}%`;
    windowEl.style.width = `${width}%`;
  }
  if (label) label.textContent = `${formatMonthDay(state.view.start)} ~ ${formatMonthDay(state.view.end)}`;
  const viewText = $("#viewText");
  if (viewText) viewText.textContent = `窗口：${state.view.start} ~ ${state.view.end}（${state.view.total} 天）`;
}

function slotCenterPct(offset, total) {
  return ((offset + 0.5) / Math.max(1, total)) * 100;
}

function slotStartPct(offset, total) {
  return (offset / Math.max(1, total)) * 100;
}

function renderGantt(tasks) {
  if (!tasks.length) {
    $("#gantt").innerHTML = `<p class="empty">当前时间窗口内没有符合筛选条件的任务。</p>`;
    return;
  }
  const { start, end, total } = state.timeline;
  $("#gantt").innerHTML = tasks.map((task) => {
    const bars = taskRenderSegments(task).map((segment, index) => {
      if (!datesOverlap(segment.start_date, segment.end_date, start, end)) return "";
      const clippedStart = maxDate(segment.start_date, start);
      const clippedEnd = minDate(segment.end_date, end);
      const left = daysBetween(start, clippedStart) / total * 100;
      const width = Math.max(1.2, (daysBetween(clippedStart, clippedEnd) + 1) / total * 100);
      const titleSuffix = segment.auto_extended ? "；自动延长至今日，交付件完备后停止延长" : "；编辑模式下拖动移动，边缘拉伸";
      return `<span class="bar" data-risk="${task.risk}" data-status="${statusClass(evaluateTaskDelivery(task).status)}" data-auto-extended="${segment.auto_extended ? "true" : "false"}" data-task-id="${escapeAttr(task.id)}" data-segment-index="${segment.source_index ?? index}" style="left:${left}%;width:${width}%" title="${escapeHtml(displayTaskTitle(task))}：${escapeHtml(segment.start_date)} ~ ${escapeHtml(segment.end_date)}${titleSuffix}"><small>${escapeHtml(formatMonthDay(segment.end_date))}</small></span>`;
    }).join("");
    const saved = taskEarlySavedSegments(task).map((segment) => {
      if (!datesOverlap(segment.start_date, segment.end_date, start, end)) return "";
      const clippedStart = maxDate(segment.start_date, start);
      const clippedEnd = minDate(segment.end_date, end);
      const left = daysBetween(start, clippedStart) / total * 100;
      const width = Math.max(1.2, (daysBetween(clippedStart, clippedEnd) + 1) / total * 100);
      return `<span class="saved-segment" style="left:${left}%;width:${width}%" title="${escapeAttr(displayTaskTitle(task))}：提前完成释放 ${escapeAttr(segment.start_date)} ~ ${escapeAttr(segment.end_date)}"><small>${escapeHtml(formatMonthDay(segment.end_date))}</small></span>`;
    }).join("");
    const marker = taskRecommitMarkerHtml(task, start, total);
    return `<div class="gantt-row"><div class="gantt-title">${escapeHtml(displayTaskTitle(task))}</div><div class="track">${bars}${saved}${marker}</div></div>`;
  }).join("");
  document.querySelectorAll(".bar").forEach((bar) => {
    bar.addEventListener("dblclick", () => {
      const task = state.data.tasks.find((item) => item.id === bar.dataset.taskId);
      if (canScheduleTask(task)) splitTask(bar.dataset.taskId);
    });
    attachGanttDrag(bar);
  });
}

function renderPeopleView(tasks) {
  const days = dateList(state.view.start, state.view.end);
  const people = peopleForTasks(tasks);
  const lanePlans = new Map(people.map((person) => [person.id, peopleLanePlan(tasks, person)]));
  if (!people.length) {
    $("#peopleView").innerHTML = `<p class="empty">当前时间窗口内没有符合筛选条件的人力安排。</p>`;
    return;
  }
  $("#peopleView").innerHTML = `
    <div class="view-note">按当前时间窗口展示每日人力占用；同一天多个事项会叠放显示。任务多的人员靠前，空闲人员靠后，待排人力不展示。</div>
    <div class="people-grid-wrap">
      <table class="people-grid">
        <thead>
          <tr>
            <th class="people-owner-head">人员</th>
            ${days.map((day) => {
              const workday = chinaWorkdayInfo(day);
              const className = workday.nonWorking ? " class=\"non-working-day\"" : "";
              return `<th${className} title="${escapeAttr(workday.label)}"><strong>${formatMonthDay(day)}</strong><small>${weekdayName(day)}${workday.nonWorking ? ` · ${escapeHtml(workday.label)}` : ""}</small></th>`;
            }).join("")}
          </tr>
        </thead>
        <tbody>
          ${people.map((person) => `
            <tr>
              <th class="people-owner">${personChipHtml(person)}</th>
              ${days.map((day) => {
                const workday = chinaWorkdayInfo(day);
                const className = workday.nonWorking ? " class=\"non-working-day\"" : "";
                return `<td${className} title="${escapeAttr(workday.label)}">${peopleLaneCellHtml(lanePlans.get(person.id), day)}</td>`;
              }).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function peopleLanePlan(tasks, person) {
  const days = dateList(state.view.start, state.view.end);
  const assignments = tasks
    .map((task) => {
      const activeDays = days.filter((day) => taskPeople(task, day).some((item) => item.id === person.id)
        && taskRenderSegments(task).some((segment) => segment.start_date <= day && segment.end_date >= day));
      return activeDays.length ? { task, start: activeDays[0], end: activeDays.at(-1) } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end) || displayTaskTitle(a.task).localeCompare(displayTaskTitle(b.task), "zh-CN"));
  const laneEnds = [];
  const taskLane = new Map();
  const taskById = new Map();
  assignments.forEach((item) => {
    let lane = laneEnds.findIndex((end) => end < item.start);
    if (lane < 0) lane = laneEnds.length;
    laneEnds[lane] = item.end;
    taskLane.set(item.task.id, lane);
    taskById.set(item.task.id, item.task);
  });
  return { personId: person.id, taskLane, taskById, laneCount: laneEnds.length };
}

function peopleLaneCellHtml(plan, day) {
  const laneCount = Math.max(1, plan?.laneCount || 0);
  const slots = Array.from({ length: laneCount }, () => `<span class="work-lane-placeholder"></span>`);
  if (!plan) return slots.join("");
  const taskItems = [...plan.taskById.values()]
    .map((task) => {
      if (!taskPeople(task, day).some((person) => person.id === plan.personId)) return null;
      const segment = taskRenderSegments(task).find((item) => item.start_date <= day && item.end_date >= day);
      return segment ? { task, segment } : null;
    })
    .filter(Boolean);
  taskItems.forEach(({ task, segment }) => {
    const lane = plan.taskLane.get(task.id);
    slots[lane] = taskChipHtml(task, segment);
  });
  return slots.join("");
}

function renderOperatorView(tasks) {
  const rows = operatorRows(tasks);
  if (!rows.length) {
    $("#operatorView").innerHTML = `<p class="empty">当前时间窗口内没有符合筛选条件的算子事项。</p>`;
    return;
  }
  const { start, end, total } = state.timeline;
  $("#operatorView").innerHTML = `
    <div class="view-note">按源仓算子全名聚合；每个算子一行，行内展示该算子的多个特性交付条。</div>
    <div class="operator-gantt">
      ${rows.map((row) => {
        const bars = row.items.map((item, index) => taskRenderSegments(item.task).map((segment, segmentIndex) => {
          if (!datesOverlap(segment.start_date, segment.end_date, start, end)) return "";
          const clippedStart = maxDate(segment.start_date, start);
          const clippedEnd = minDate(segment.end_date, end);
          const left = daysBetween(start, clippedStart) / total * 100;
          const width = Math.max(2, (daysBetween(clippedStart, clippedEnd) + 1) / total * 100);
          const isDelaySegment = Boolean(segment.auto_extended);
          const titleSuffix = isDelaySegment ? " · 自动延长段，交付件完备后停止延长" : "";
          return `
            <span class="operator-bar ${isDelaySegment ? "delay-chip" : riskClass(item.task.risk)}" style="left:${left}%;width:${width}%;top:${index * 26 + 6}px" title="${escapeAttr(displayTaskTitle(item.task))} · ${escapeAttr(segment.start_date)} ~ ${escapeAttr(segment.end_date)}${titleSuffix}">
              ${isDelaySegment ? `<small>${escapeHtml(formatMonthDay(segment.end_date))}</small>` : `<b>${escapeHtml(featureTitle(item.task, item.operator))}</b><small>${escapeHtml(formatMonthDay(segment.end_date))}</small>`}
            </span>
          `;
        }).join("")).join("");
        return `
          <div class="operator-gantt-row" style="--lane-count:${Math.max(1, row.items.length)}">
            <div class="operator-name">
              <strong>${escapeHtml(row.operator.label)}</strong>
              <span>${row.items.length} 项</span>
            </div>
            <div class="operator-track">${bars}</div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function operatorRows(tasks) {
  const rows = new Map();
  tasks.forEach((task) => {
    taskOperators(task).forEach((operator) => {
      if (!rows.has(operator.id)) rows.set(operator.id, { operator, items: [] });
      rows.get(operator.id).items.push({ task, operator });
    });
  });
  return [...rows.values()].sort((a, b) => a.operator.label.localeCompare(b.operator.label, "zh-CN"))
    .map((row) => ({
      ...row,
      items: [...row.items].sort((a, b) => a.task.end_date.localeCompare(b.task.end_date) || displayTaskTitle(a.task).localeCompare(displayTaskTitle(b.task), "zh-CN")),
    }));
}

function tasksForPersonOnDay(tasks, person, day) {
  return tasks.filter((task) => taskPeople(task, day).some((item) => item.id === person.id) && taskRenderSegments(task).some((segment) => segment.start_date <= day && segment.end_date >= day));
}

function tasksForPerson(person) {
  return (state.data.tasks || []).filter((task) => taskOwnerNames(task).includes(person.name));
}

function taskChipHtml(task, segment = null) {
  const operators = taskOperators(task);
  const operatorLabel = operators.length ? operators.map((operator) => operator.label).join(" / ") : specialTitle(task.special_id);
  const isDelaySegment = Boolean(segment?.auto_extended);
  const titleSuffix = isDelaySegment ? "；自动延长段，交付件完备后停止延长" : "";
  return `<span class="work-chip ${isDelaySegment ? "delay-chip" : riskClass(task.risk)}" title="${escapeAttr(displayTaskTitle(task))}${titleSuffix}">
    <em>${escapeHtml(operatorLabel || groupTitle(task.group_id))}</em>
    <b>${escapeHtml(compactTaskTitle(task))}</b>
  </span>`;
}

function taskOperatorLabelText(task) {
  return taskOperators(task).map((operator) => operator.label).join(" / ");
}

function operatorChipsHtml(task) {
  const labels = taskOperators(task).map((operator) => operator.label);
  return labels.length
    ? labels.map((label) => `<span class="operator-chip">${escapeHtml(label)}</span>`).join("")
    : `<span class="muted-cell">-</span>`;
}

function featureTitle(task, operator = taskOperators(task)[0]) {
  let title = task.title || "";
  if (title.includes("fwd_h 与 fwd_o")) title = title.replace("fwd_h 与 fwd_o", "");
  if (title.startsWith("多算子")) return title;
  const aliases = operator?.aliases || [];
  const matched = aliases.find((alias) => title.toLowerCase().startsWith(alias.toLowerCase()));
  if (matched) title = title.slice(matched.length);
  title = title.replace(/^新增\s*/, "").replace(/^算子\s*/, "").trim();
  return title || displayTaskTitle(task);
}

function taskOperators(task) {
  const explicitIds = parseOperatorIds(task?.operator_ids);
  return explicitIds.map(operatorById).filter(Boolean);
}

function operatorById(id) {
  return operatorCatalog().find((operator) => operator.id === id);
}

function operatorOwnerName(operatorId, referenceDate) {
  const rules = operatorById(operatorId)?.owner_rules || OPERATOR_OWNER_RULES[operatorId] || [];
  if (!rules.length) return "";
  return (rules.find((rule) => !rule.until || !referenceDate || referenceDate <= rule.until) || rules[rules.length - 1]).owner || "";
}

function operatorOwnerNamesForTask(task, referenceDate = "") {
  const operators = task ? taskOperators(task) : [];
  if (!operators.length) return [];
  const dates = referenceDate ? [referenceDate] : ownerReferenceDates(task);
  return uniqueStrings(operators.flatMap((operator) => dates.map((date) => operatorOwnerName(operator.id, date))));
}

function ownerReferenceDates(task) {
  const dates = [];
  taskSegments(task).forEach((segment) => {
    if (segment.start_date) dates.push(segment.start_date);
    if (segment.end_date) dates.push(segment.end_date);
    operatorCatalog().flatMap((operator) => operator.owner_rules || []).forEach((rule) => {
      if (rule.until && segment.start_date <= rule.until && segment.end_date > rule.until) dates.push(addDays(rule.until, 1));
    });
  });
  return uniqueStrings(dates.length ? dates : [task?.start_date || state.view.start]).filter(Boolean).sort();
}

function operatorCatalog() {
  const source = Array.isArray(state.data?.operators) && state.data.operators.length ? state.data.operators : defaultOperators();
  return source.map(normalizeOperator).filter((operator) => operator.active !== false);
}

function ensureOperatorCatalog() {
  const source = Array.isArray(state.data?.operators) && state.data.operators.length ? state.data.operators : defaultOperators();
  state.data.operators = source.map(normalizeOperator)
    .sort((a, b) => Number(a.position || 0) - Number(b.position || 0) || a.label.localeCompare(b.label, "zh-CN"));
}

function defaultOperators() {
  return OPERATOR_RULES.map((operator, index) => ({
    ...operator,
    owner_rules: OPERATOR_OWNER_RULES[operator.id] || [],
    position: index,
    active: true,
  }));
}

function normalizeOperator(operator) {
  return {
    id: String(operator.id || "").trim(),
    label: String(operator.label || operator.id || "").trim(),
    aliases: normalizeOperatorAliases(operator.aliases || [operator.label || operator.id]),
    owner_rules: normalizeOperatorOwnerRules(operator.owner_rules || operator.ownerRules || OPERATOR_OWNER_RULES[operator.id] || []),
    position: Number.isFinite(Number(operator.position)) ? Number(operator.position) : 0,
    active: operator.active === false || operator.active === 0 || operator.active === "0" ? false : true,
  };
}

function normalizeOperatorAliases(value) {
  const raw = Array.isArray(value) ? value : String(value || "").split(/[、,，;；\n]+/);
  return uniqueStrings(raw.map((item) => String(item || "").trim()).filter(Boolean));
}

function normalizeOperatorOwnerRules(value) {
  const raw = Array.isArray(value) ? value : parseJsonValue(value, []);
  return raw.map((rule) => ({
    ...(rule.until ? { until: String(rule.until).trim() } : {}),
    owner: normalizeOwnerName(rule.owner || ""),
  })).filter((rule) => rule.owner);
}

function parseOperatorIds(value) {
  return uniqueStrings(String(value || "")
    .split(/[、/,，;；\s]+/)
    .map((item) => item.trim())
    .filter(Boolean));
}

function normalizeOperatorIdsText(value) {
  return parseOperatorIds(value).join("/");
}

function normalizeOperatorId(value) {
  return String(value || "")
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^\w.-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function operatorLabelsForIds(value) {
  return parseOperatorIds(value).map((id) => operatorById(id)?.label || id).filter(Boolean);
}

function formatOperatorOwnerRules(rules) {
  const normalized = normalizeOperatorOwnerRules(rules);
  return normalized.length
    ? normalized.map((rule) => `${rule.until ? `${rule.until} 前 ` : ""}${rule.owner}`).join("；")
    : "未设置";
}

function operatorOwnerRulesInputText(rules) {
  return normalizeOperatorOwnerRules(rules)
    .map((rule) => `${rule.until || "*"}:${rule.owner}`)
    .join(";");
}

function parseOperatorOwnerRulesInput(value) {
  return String(value || "").split(/[;；\n]+/).map((chunk) => {
    const text = chunk.trim();
    if (!text) return null;
    const match = text.match(/^(\d{4}-\d{2}-\d{2}|\*)\s*[:：]\s*(.+)$/);
    if (!match) return { owner: normalizeOwnerName(text) };
    const owner = normalizeOwnerName(match[2]);
    return match[1] === "*" ? { owner } : { until: match[1], owner };
  }).filter((rule) => rule?.owner);
}

function parseJsonValue(value, fallback) {
  if (Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(String(value || ""));
    return Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function displayTaskTitle(task) {
  const operators = taskOperators(task);
  if (!operators.length) return task.title || "";
  let title = task.title || "";
  const placeholders = [];
  operators.forEach((operator) => {
    operator.aliases
      .slice()
      .sort((a, b) => b.length - a.length)
      .forEach((alias) => {
        title = title.replace(new RegExp(escapeRegExp(alias), "ig"), () => {
          const key = `__OP_${placeholders.length}__`;
          placeholders.push([key, operator.label]);
          return key;
        });
      });
  });
  placeholders.forEach(([key, label]) => { title = title.replaceAll(key, label); });
  title = title.replace(/prepare_wy_repr_bwd_full\s+prepare_wy_repr_bwd_full/ig, "prepare_wy_repr_bwd_full");
  return title;
}

function compactTaskTitle(task) {
  const operators = taskOperators(task);
  if (!operators.length) return task.title || "";
  if (operators.length > 1) return featureTitle(task, operators[0]);
  return featureTitle(task, operators[0]);
}

function ensurePeopleCatalog() {
  const existing = new Map((state.data.people || []).map((person) => [person.name, person]));
  const names = ["待排人力"];
  (state.data.people || []).forEach((person) => {
    const name = normalizeOwnerName(person.name);
    if (!names.includes(name)) names.push(name);
  });
  operatorOwnerRuleNames().forEach((name) => {
    if (!names.includes(name)) names.push(name);
  });
  (state.data.tasks || []).forEach((task) => taskOwnerNames(task).forEach((name) => {
    if (!names.includes(name)) names.push(name);
  }));
  const people = [];
  names.forEach((name, index) => {
    const current = existing.get(name);
    people.push({
      id: current?.id || `P${String(index + 1).padStart(2, "0")}`,
      name,
      position: Number.isFinite(current?.position) ? current.position : index,
      pl: normalizePl(current?.pl),
      placeholder: isPlaceholderOwner(name),
    });
  });
  state.data.people = people.sort(comparePeople);
}

function normalizeOwnerName(name) {
  const value = String(name || "").trim();
  return !value || value === "待填写" || value === "待排人力" ? "待排人力" : value;
}

function splitOwnerNames(owner, task = null, referenceDate = "") {
  const raw = normalizeOwnerName(owner);
  return uniqueStrings(raw.split(/[、/,，;；&]+/).flatMap((name) => {
    const normalized = normalizeOwnerName(name);
    if (normalized !== "对应算子责任人") return [normalized];
    const owners = operatorOwnerNamesForTask(task, referenceDate);
    return owners.length ? owners : [normalized];
  }).filter(Boolean));
}

function taskOwnerNames(task, referenceDate = "") {
  return splitOwnerNames(task?.owner, task, referenceDate);
}

function taskPeople(task, referenceDate = "") {
  const byName = new Map((state.data.people || []).map((person) => [person.name, person]));
  return taskOwnerNames(task, referenceDate).map((name) => {
    const person = byName.get(name);
    return person ? { ...person, pl: normalizePl(person.pl) } : { id: "P??", name, position: 9999, pl: DEFAULT_PL, placeholder: isPlaceholderOwner(name) };
  });
}

function peopleForTasks(tasks) {
  ensurePeopleCatalog();
  const visiblePerson = (person) => !isPlaceholderOwner(person.name);
  const selectedOwners = ownerFilterValues();
  if (selectedOwners.length) {
    const selected = new Set(selectedOwners);
    return (state.data.people || [])
      .filter((person) => visiblePerson(person) && selected.has(person.name))
      .sort((a, b) => comparePeopleForView(a, b, tasks));
  }
  if (state.filters.pl) {
    const selectedPl = normalizePl(state.filters.pl);
    const people = uniqueBy([
      ...(state.data.people || []).filter((person) => normalizePl(person.pl) === selectedPl),
      ...tasks.flatMap(taskPeople).filter((person) => normalizePl(person.pl) === selectedPl),
    ], "id").filter(visiblePerson);
    return people.sort((a, b) => comparePeopleForView(a, b, tasks));
  }
  const basePeople = isDeveloperEditMode() ? [] : (state.data.people || []).filter(visiblePerson);
  const people = uniqueBy([
    ...basePeople,
    ...tasks.flatMap(taskPeople).filter(visiblePerson),
  ], "id");
  return people.sort((a, b) => comparePeopleForView(a, b, tasks));
}

function comparePeople(a, b) {
  const waitingA = a.name === "待排人力" ? 0 : 1;
  const waitingB = b.name === "待排人力" ? 0 : 1;
  return waitingA - waitingB || a.position - b.position || a.id.localeCompare(b.id);
}

function comparePeopleForView(a, b, tasks) {
  const countA = personTaskCountInView(a, tasks);
  const countB = personTaskCountInView(b, tasks);
  return countB - countA || comparePeople(a, b);
}

function personIsIdleInView(person, tasks) {
  return personTaskCountInView(person, tasks) === 0;
}

function personTaskCountInView(person, tasks) {
  const days = dateList(state.view.start, state.view.end);
  return tasks.filter((task) => days.some((day) => taskPeople(task, day).some((item) => item.id === person.id)
    && taskRenderSegments(task).some((segment) => segment.start_date <= day && segment.end_date >= day))).length;
}

function personChipHtml(person) {
  const klass = person.placeholder ? " placeholder" : "";
  return `<span class="person-chip${klass}"><em>${escapeHtml(person.name)}</em></span>`;
}

function ownerChipsHtml(task) {
  const chips = taskPeople(task).map(personChipHtml).join("");
  return `<span class="owner-chip-list">${chips}</span>`;
}

function normalizePl(value) {
  const text = String(value || "").trim();
  return PL_OPTIONS.includes(text) ? text : DEFAULT_PL;
}

function normalizeEmail(value) {
  return String(value || "").trim();
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizeEmail(value));
}

function taskPlNames(task, referenceDate = "") {
  return uniqueStrings(taskPeople(task, referenceDate).map((person) => normalizePl(person.pl)));
}

function taskPlChipsHtml(task) {
  return taskPlNames(task).map((pl) => `<span class="pl-chip">${escapeHtml(pl)}</span>`).join("");
}

function plSelectHtml(value) {
  const current = normalizePl(value);
  return `<select data-person-pl>${PL_OPTIONS.map((pl) => `<option value="${escapeAttr(pl)}" ${pl === current ? "selected" : ""}>${escapeHtml(pl)}</option>`).join("")}</select>`;
}

function ownerPickerOptions(task) {
  return uniqueStrings([
    "待排人力",
    ...(state.data.people || []).map((person) => person.name),
    ...(state.data.tasks || []).flatMap((item) => taskOwnerNames(item)),
    ...taskOwnerNames(task),
    ...operatorOwnerRuleNames(),
  ])
    .filter((name) => name && !/[、/,，;；&]/.test(name))
    .sort((a, b) => {
      if (a === "待排人力") return -1;
      if (b === "待排人力") return 1;
      return a.localeCompare(b, "zh-CN");
    });
}

function ownerEditorHtml(task) {
  const owners = taskOwnerNames(task);
  const options = ownerPickerOptions(task).map((name) => `
    <label class="owner-option">
      <input type="checkbox" data-owner-picker-value value="${escapeAttr(name)}" ${owners.includes(name) ? "checked" : ""}>
      <span>${escapeHtml(name)}</span>
    </label>
  `).join("");
  return `
    <div class="owner-editor">
      <div class="owner-combo">
        <input class="owner-input" data-field="owner" value="${escapeAttr(task.owner)}" placeholder="可手动输入，或勾选责任人">
        <button type="button" class="owner-picker-toggle" data-owner-picker-toggle>选择</button>
      </div>
      <div class="owner-picker-panel">${options}</div>
      <div class="owner-hint">勾选后自动写入责任人字段；也可直接手动编辑</div>
      <div class="owner-preview">${ownerChipsHtml(task)}</div>
    </div>
  `;
}

function operatorEditorHtml(task) {
  const selected = parseOperatorIds(task.operator_ids);
  const selectedSet = new Set(selected);
  const options = operatorCatalog().map((operator) => `
    <label class="owner-option">
      <input type="checkbox" data-operator-picker-value value="${escapeAttr(operator.id)}" ${selectedSet.has(operator.id) ? "checked" : ""}>
      <span>${escapeHtml(operator.label)}</span>
    </label>
  `).join("");
  const hint = selected.length
    ? "已显式关联算子，可多选；不选则不强制关联"
    : "未关联算子，可保持为空";
  return `
    <div class="operator-editor">
      <input type="hidden" data-field="operator_ids" value="${escapeAttr(normalizeOperatorIdsText(task.operator_ids))}">
      <button type="button" class="owner-picker-toggle" data-operator-picker-toggle>选择算子</button>
      <div class="owner-picker-panel">${options}</div>
      <div class="owner-hint">${escapeHtml(hint)}</div>
      <div class="operator-preview">${operatorChipsHtml(task)}</div>
    </div>
  `;
}

function toggleOwnerPicker(button) {
  const editor = button.closest(".owner-editor");
  if (!editor) return;
  const willOpen = !editor.classList.contains("open");
  closeOwnerPickers(null, editor);
  editor.classList.toggle("open", willOpen);
  if (willOpen) positionOwnerPicker(editor);
}

function positionOwnerPicker(editor) {
  const panel = editor?.querySelector(".owner-picker-panel");
  const tableWrap = editor?.closest(".table-wrap");
  if (!panel || !tableWrap) return;
  editor.classList.remove("drop-up");
  const editorRect = editor.getBoundingClientRect();
  const wrapRect = tableWrap.getBoundingClientRect();
  const defaultMaxHeight = 220;
  const gap = 8;
  const below = Math.max(40, wrapRect.bottom - (editorRect.top + 40) - gap);
  const above = Math.max(40, editorRect.top - wrapRect.top - gap);
  const dropUp = below < 160 && above > below;
  editor.classList.toggle("drop-up", dropUp);
  panel.style.maxHeight = `${Math.min(defaultMaxHeight, dropUp ? above : below)}px`;
}

function closeOwnerPickers(event = null, except = null) {
  if (event?.target?.closest?.(".owner-picker-panel")) return;
  document.querySelectorAll(".owner-editor.open,.operator-editor.open").forEach((editor) => {
    if (editor === except) return;
    editor.classList.remove("open", "drop-up");
    const panel = editor.querySelector(".owner-picker-panel");
    if (panel) panel.style.maxHeight = "";
  });
}

function toggleOperatorPicker(button) {
  const editor = button.closest(".operator-editor");
  if (!editor) return;
  const willOpen = !editor.classList.contains("open");
  closeOwnerPickers(null, editor);
  editor.classList.toggle("open", willOpen);
  if (willOpen) positionOwnerPicker(editor);
}

function syncOperatorFromPicker(control) {
  const row = control.closest("tr");
  const input = row?.querySelector('[data-field="operator_ids"]');
  if (!input) return;
  const selected = [...row.querySelectorAll("[data-operator-picker-value]:checked")].map((option) => option.value).filter(Boolean);
  input.value = selected.join("/");
  markTaskDirty(input);
  syncOperatorPreview(row);
}

function syncOperatorPreview(row) {
  const task = state.data.tasks.find((item) => item.id === row?.dataset.taskId);
  const preview = row?.querySelector(".operator-preview");
  if (task && preview) preview.innerHTML = operatorChipsHtml(task);
}

function syncOwnerFromPicker(control) {
  const row = control.closest("tr");
  const input = row?.querySelector('[data-field="owner"]');
  if (!input) return;
  const selected = [...row.querySelectorAll("[data-owner-picker-value]:checked")].map((option) => option.value).filter(Boolean);
  input.value = selected.length ? selected.join("/") : "待排人力";
  markTaskDirty(input);
  syncOwnerPreview(row);
}

function syncOwnerPickerFromInput(input) {
  const row = input.closest("tr");
  const names = splitOwnerNames(input.value);
  row?.querySelectorAll("[data-owner-picker-value]").forEach((option) => { option.checked = names.includes(option.value); });
  syncOwnerPreview(row);
}

function syncOwnerPreview(row) {
  const task = state.data.tasks.find((item) => item.id === row?.dataset.taskId);
  const preview = row?.querySelector(".owner-preview");
  if (task && preview) preview.innerHTML = ownerChipsHtml(task);
}

function isPlaceholderOwner(name) {
  return /待填|待排|对应/.test(name);
}

function uniqueBy(items, field) {
  const seen = new Set();
  return items.filter((item) => {
    const key = item[field];
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function uniqueStrings(items) {
  return [...new Set(items.filter(Boolean))];
}

function buildChinaWorkCalendar(holidayRanges, adjustedWorkdays) {
  const holidays = {};
  holidayRanges.forEach(([name, start, end]) => {
    dateList(start, end).forEach((day) => { holidays[day] = name; });
  });
  return {
    holidays,
    adjustedWorkdays: Object.fromEntries(adjustedWorkdays),
  };
}

function chinaWorkdayInfo(value) {
  if (!isYmd(value)) return { nonWorking: false, label: "" };
  const calendar = CHINA_WORK_CALENDARS[value.slice(0, 4)];
  if (calendar?.adjustedWorkdays?.[value]) {
    return { nonWorking: false, adjustedWorkday: true, label: calendar.adjustedWorkdays[value] };
  }
  if (calendar?.holidays?.[value]) {
    return { nonWorking: true, holiday: true, label: calendar.holidays[value] };
  }
  if (isWeekend(value)) {
    return { nonWorking: true, weekend: true, label: "周末" };
  }
  return { nonWorking: false, label: "工作日" };
}

function workdaysUntil(start, end) {
  if (!isYmd(start) || !isYmd(end) || end <= start) return 0;
  let count = 0;
  for (let day = addDays(start, 1); day <= end; day = addDays(day, 1)) {
    if (!chinaWorkdayInfo(day).nonWorking) count += 1;
  }
  return count;
}

function isWeekend(value) {
  const day = dateFromYmd(value).getUTCDay();
  return day === 0 || day === 6;
}

function dateList(start, end) {
  const result = [];
  for (let day = start; day <= end; day = addDays(day, 1)) result.push(day);
  return result;
}

function weekdayName(value) {
  return ["日", "一", "二", "三", "四", "五", "六"][dateFromYmd(value).getUTCDay()];
}

function dateCellHtml(value) {
  return isYmd(value)
    ? `<span class="date-pill">${escapeHtml(value)}</span>`
    : `<span class="muted-cell">-</span>`;
}

function linkListHtml(value, label) {
  if (label === "PR") return prLinkPillsHtml(value);
  const links = parseLinks(value);
  if (!links.length) return `<span class="muted-cell">-</span>`;
  return links.map((link, index) => `<a class="link-pill" href="${escapeAttr(link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}${links.length > 1 ? index + 1 : ""}</a>`).join("");
}

function parseLinks(value) {
  return String(value || "").split(/[\s,，;；]+/).map((item) => item.trim()).filter((item) => /^https?:\/\//i.test(item));
}

function parsePrRefs(value) {
  return String(value || "").split(/[\s,，;；]+/)
    .map((item) => item.trim())
    .filter((item) => item && (/^https?:\/\//i.test(item) || /^#?\d+$/.test(item)));
}

function prLinkSummary(value) {
  const refs = parsePrRefs(value);
  const matches = refs.map((ref) => findPrCandidate(ref));
  const missing = !refs.length || matches.some((item) => !item);
  return {
    refs,
    matches: matches.filter(Boolean),
    missing,
    allMerged: refs.length > 0 && !missing && matches.every((item) => item.status === "merged"),
    hasOpen: refs.length > 0 && !missing && matches.some((item) => item.status === "open"),
  };
}

function taskHasReport(task) {
  return Boolean(String(task.test_report || "").trim());
}

function taskRequiresPr(task) {
  return normalizeBooleanFlag(task?.pr_required, true) === 1;
}

function taskIsCompletionOverride(task) {
  return /ops\s*目录整改/i.test(String(task.title || ""));
}

function taskDdl(task) {
  return isYmd(task.end_date) ? task.end_date : (isYmd(task.start_date) ? task.start_date : todayBjYmd());
}

function todayBjYmd() {
  return new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString().slice(0, 10);
}

function taskWorkdaysUntilDdl(task) {
  return workdaysUntil(todayBjYmd(), taskDdl(task));
}

function taskDdlWorkdayDelta(task) {
  const today = todayBjYmd();
  const ddl = taskDdl(task);
  if (evaluateTaskStatus(task) === "delayed") {
    return -Math.max(1, workdaysUntil(ddl, today));
  }
  return workdaysUntil(today, ddl);
}

function taskDdlCountdownHtml(task) {
  const ddl = taskDdl(task);
  const delta = taskDdlWorkdayDelta(task);
  const stateClass = delta < 0 ? "overdue" : (delta === 0 ? "today" : "upcoming");
  const label = `${delta > 0 ? "+" : ""}${delta} 工作日`;
  const title = `DDL：${ddl}；今天：${todayBjYmd()}；按中国法定节假日、调休工作日和周末计算`;
  return `
    <span class="ddl-countdown ${stateClass}" title="${escapeAttr(title)}">
      <strong>${escapeHtml(label)}</strong>
      <small>DDL ${escapeHtml(formatMonthDay(ddl))}</small>
    </span>
  `;
}

function taskPastDdlMidnight(task) {
  return todayBjYmd() > taskDdl(task);
}

function taskHasWaitingOwner(task) {
  return taskOwnerNames(task).includes("待排人力");
}

function taskHasClosedSchedule(task) {
  return isYmd(task.start_date) && isYmd(task.end_date);
}

function evaluateTaskRisk(task) {
  const pr = prLinkSummary(task.pr_link);
  const workdaysUntilDdl = taskWorkdaysUntilDdl(task);
  if (taskHasWaitingOwner(task)) {
    return "高";
  }
  if (!taskRequiresPr(task)) {
    return taskHasReport(task) ? "低" : (workdaysUntilDdl <= 6 ? "高" : "中");
  }
  if (pr.allMerged) {
    return "低";
  }
  if (pr.hasOpen) {
    return workdaysUntilDdl <= 3 ? "中" : "低";
  }
  return workdaysUntilDdl <= 6 ? "高" : "中";
}

function evaluateTaskStatus(task) {
  const pr = prLinkSummary(task.pr_link);
  const completed = taskIsCompletionOverride(task) || (taskHasReport(task) && (!taskRequiresPr(task) || pr.allMerged));
  if (completed) {
    return "done";
  }
  if (taskHasWaitingOwner(task) || !taskHasClosedSchedule(task)) {
    return "todo";
  }
  if (task.status === "blocked") {
    return "blocked";
  }
  if (taskPastDdlMidnight(task)) {
    return "delayed";
  }
  return "doing";
}

function evaluateTaskDelivery(task) {
  return { risk: evaluateTaskRisk(task), status: evaluateTaskStatus(task) };
}

function taskVisibleOnTimeline(task) {
  return evaluateTaskStatus(task) !== "todo";
}

function syncTaskDeliveryRules(task) {
  const previousStatus = task.status;
  const next = evaluateTaskDelivery(task);
  const changed = [];
  if (task.risk !== next.risk) {
    changed.push("risk");
    task.risk = next.risk;
  }
  if (task.status !== next.status) {
    changed.push("status");
    task.status = next.status;
  }
  if (next.status !== "done" && task.done_date) {
    changed.push("done_date");
    task.done_date = "";
  } else if (next.status === "done" && previousStatus !== "done" && !isYmd(task.done_date)) {
    changed.push("done_date");
    task.done_date = todayBjYmd();
  }
  return changed;
}

function prLinkEditorHtml(task) {
  return `
    <div class="pr-link-editor">
      <input class="link-input" data-field="pr_link" placeholder="PR URL，可填多个" value="${escapeAttr(task.pr_link || "")}">
      ${prCatalogQuickPickerHtml()}
      ${prLinkPreviewHtml(task.pr_link)}
    </div>
  `;
}

function prLinkPreviewHtml(value) {
  const html = prLinkPillsHtml(value);
  return html.includes("muted-cell") ? "" : `<div class="pr-link-preview">${html}</div>`;
}

function prLinkPillsHtml(value) {
  const refs = parsePrRefs(value);
  if (!refs.length) return `<span class="muted-cell">-</span>`;
  return refs.map((ref, index) => prPillHtml(ref, index, refs.length)).join("");
}

function prPillHtml(ref, index, total) {
  const pr = findPrCandidate(ref);
  const status = pr?.status === "merged" ? "merged" : (pr?.status === "open" ? "open" : "unknown");
  const href = pr?.url || (/^https?:\/\//i.test(ref) ? ref : "");
  const label = total > 1 ? `PR${index + 1}` : "PR";
  const title = pr ? `#${pr.number} ${pr.title || ""}`.trim() : String(ref || "");
  const body = escapeHtml(label);
  const attrs = `class="link-pill pr-${status}" title="${escapeAttr(title)}"`;
  if (!href) return `<span ${attrs}>${body}</span>`;
  return `<a ${attrs} href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${body}</a>`;
}

function prCatalogQuickPickerHtml() {
  const items = state.prCatalog?.items || [];
  if (!items.length) {
    return `
      <div class="pr-quick-picker">
        <input class="pr-search" data-pr-search disabled placeholder="暂无 PR 候选">
        <button type="button" data-pr-append disabled>追加</button>
        <div class="pr-suggest-menu" data-pr-suggest-menu></div>
      </div>
    `;
  }
  return `
    <div class="pr-quick-picker">
      <input class="pr-search" data-pr-search autocomplete="off" placeholder="输入 PR 号 / 标题 / 链接，回车追加">
      <button type="button" data-pr-append>追加</button>
      <div class="pr-suggest-menu" data-pr-suggest-menu></div>
    </div>
  `;
}

function prOptionLabel(pr) {
  return `#${pr.number} ${pr.title || ""}`.trim();
}

function sortedPrCatalogItems() {
  return [...(state.prCatalog?.items || [])].sort((a, b) => {
    const byNumber = (Number(b.number) || 0) - (Number(a.number) || 0);
    return byNumber || String(b.updated_at || b.updatedAt || "").localeCompare(String(a.updated_at || a.updatedAt || ""));
  });
}

function prSuggestionItems(query) {
  const normalized = String(query || "").trim().toLowerCase();
  return sortedPrCatalogItems().filter((pr) => {
    if (!normalized) return true;
    return [String(pr.number || ""), `#${pr.number || ""}`, pr.url, prOptionLabel(pr), pr.title, pr.headRef]
      .some((field) => String(field || "").toLowerCase().includes(normalized));
  }).slice(0, 10);
}

function prSuggestItemHtml(pr) {
  const status = pr?.status === "merged" ? "merged" : (pr?.status === "open" ? "open" : "unknown");
  return `
    <button type="button" class="pr-suggest-item pr-${status}" data-pr-suggest="${escapeAttr(pr.url || prOptionLabel(pr))}" title="${escapeAttr(prOptionLabel(pr))}">
      <strong>#${escapeHtml(pr.number)}</strong>
      <span>${escapeHtml(pr.title || "")}</span>
    </button>
  `;
}

function renderPrSuggestions(input, open = true) {
  const picker = input.closest(".pr-quick-picker");
  const menu = picker?.querySelector("[data-pr-suggest-menu]");
  if (!picker || !menu) return;
  const items = prSuggestionItems(input.value);
  menu.innerHTML = items.length
    ? items.map(prSuggestItemHtml).join("")
    : `<div class="pr-suggest-empty">无匹配 PR</div>`;
  picker.classList.toggle("open", open);
}

function closePrSuggestMenus(event = null) {
  if (event?.target?.closest?.(".pr-suggest-menu")) return;
  document.querySelectorAll(".pr-quick-picker.open").forEach((picker) => picker.classList.remove("open"));
}

function choosePrSuggestion(button) {
  const row = button.closest("tr");
  const search = row?.querySelector("[data-pr-search]");
  if (!search) return;
  search.value = button.dataset.prSuggest || "";
  appendPrFromSearch(search);
  closePrSuggestMenus();
}

function findPrCandidate(query) {
  const value = String(query || "").trim();
  if (!value) return null;
  const items = sortedPrCatalogItems();
  const normalized = value.toLowerCase();
  const number = normalized.match(/^#?(\d+)$/)?.[1]
    || normalized.match(/\/pull\/(\d+)/)?.[1]
    || normalized.match(/^#?(\d+)\b/)?.[1];
  if (number) {
    const byNumber = items.find((pr) => String(pr.number) === number);
    if (byNumber) return byNumber;
  }
  return items.find((pr) => [pr.url, prOptionLabel(pr), pr.title, pr.headRef].some((field) => String(field || "").toLowerCase().includes(normalized))) || null;
}

function appendPrFromSearch(control) {
  const row = control.closest("tr");
  const search = row?.querySelector("[data-pr-search]");
  const candidate = findPrCandidate(search?.value);
  if (!search?.value) return;
  if (!candidate) {
    alert(`未在候选池中匹配到 PR：${search.value}`);
    return;
  }
  const input = row?.querySelector('[data-field="pr_link"]');
  if (!input) return;
  const links = String(input.value || "").split(/[\s,，;；]+/).map((item) => item.trim()).filter(Boolean);
  if (!links.includes(candidate.url)) links.push(candidate.url);
  input.value = links.join(" ");
  search.value = "";
  markTaskDirty(input);
}

function readOnlyTaskRowHtml(task, className = "") {
  return `
    <tr class="${escapeAttr(className)}">
      <td><span class="tag ${riskClass(task.risk)}">${escapeHtml(task.risk)}</span></td>
      <td><span class="tag ${String(task.priority).toLowerCase()}">${escapeHtml(task.priority)}</span></td>
      <td>${escapeHtml(displayTaskTitle(task))}</td>
      <td class="col-operator">${operatorChipsHtml(task)}</td>
      <td class="col-owner">${ownerChipsHtml(task)}</td>
      <td class="col-pl">${taskPlChipsHtml(task)}</td>
      <td class="col-group">${escapeHtml(groupTitle(task.group_id))}</td>
      <td class="col-special">${escapeHtml(specialTitle(task.special_id))}</td>
      <td class="col-date">${taskDdlCountdownHtml(task)}</td>
      <td class="col-commit">${dateCellHtml(task.recommit_date)}</td>
      <td class="col-done-date">${dateCellHtml(task.done_date)}</td>
      <td class="col-pr-required">${prRequirementPillHtml(task)}</td>
      <td class="col-pr">${linkListHtml(task.pr_link, "PR")}</td>
      <td class="col-report">${linkListHtml(task.test_report, "报告")}</td>
      <td><span class="status-pill ${statusClass(task.status)}">${escapeHtml(statusLabel(task.status))}</span></td>
      <td class="edit-only"></td>
    </tr>
  `;
}

function taskActionButtonsHtml() {
  if (isDeveloperEditMode()) {
    return `<span class="ops"><button data-action="save">保存</button></span>`;
  }
  const deleteButton = isAdminEditMode() ? `<button class="danger" data-action="delete">删除</button>` : "";
  return `<span class="ops"><button data-action="save">保存</button><button data-action="split">切分</button>${deleteButton}</span>`;
}

function prRequirementPillHtml(task) {
  return taskRequiresPr(task)
    ? `<span class="requirement-pill required">需 PR</span>`
    : `<span class="requirement-pill report-only">仅报告</span>`;
}

function developerTaskRowHtml(task) {
  return `
    <tr data-task-id="${escapeAttr(task.id)}" class="${state.dirtyTaskIds.has(task.id) ? "dirty" : ""}">
      <td><span class="tag ${riskClass(task.risk)}">${escapeHtml(task.risk)}</span></td>
      <td><span class="tag ${String(task.priority).toLowerCase()}">${escapeHtml(task.priority)}</span></td>
      <td>${escapeHtml(displayTaskTitle(task))}</td>
      <td class="col-operator">${operatorChipsHtml(task)}</td>
      <td class="col-owner">${ownerChipsHtml(task)}</td>
      <td class="col-pl">${taskPlChipsHtml(task)}</td>
      <td class="col-group">${escapeHtml(groupTitle(task.group_id))}</td>
      <td class="col-special">${escapeHtml(specialTitle(task.special_id))}</td>
      <td class="col-date">${taskDdlCountdownHtml(task)}</td>
      <td class="col-commit">${dateCellHtml(task.recommit_date)}</td>
      <td class="col-done-date">${dateCellHtml(task.done_date)}</td>
      <td class="col-pr-required">${prRequirementPillHtml(task)}</td>
      <td class="col-pr">${prLinkEditorHtml(task)}</td>
      <td class="col-report"><input class="link-input" data-field="test_report" placeholder="报告 URL" value="${escapeAttr(task.test_report || "")}"></td>
      <td><span class="status-pill ${statusClass(task.status)}">${escapeHtml(statusLabel(task.status))}</span></td>
      <td class="edit-only">${taskActionButtonsHtml(task)}</td>
    </tr>
  `;
}

function renderRows(tasks) {
  const clearanceRow = tasks.length
    ? `<tr class="task-filter-clearance" aria-hidden="true"><td colspan="16"></td></tr>`
    : "";
  $("#rows").innerHTML = clearanceRow + tasks.map((task) => {
    if (!state.token) {
      return readOnlyTaskRowHtml(task);
    }
    if (!canEditTask(task)) {
      return readOnlyTaskRowHtml(task, "readonly-related");
    }
    if (isDeveloperEditMode()) {
      return developerTaskRowHtml(task);
    }
    return `
      <tr data-task-id="${escapeAttr(task.id)}" class="${state.dirtyTaskIds.has(task.id) ? "dirty" : ""}">
        <td>${selectHtml("risk", [["高","高"],["中","中"],["低","低"]], task.risk)}</td>
        <td>${selectHtml("priority", [["P0","P0"],["P1","P1"],["P2","P2"]], task.priority)}</td>
        <td><input class="title-input" data-field="title" value="${escapeAttr(task.title)}"></td>
        <td class="col-operator">${operatorEditorHtml(task)}</td>
        <td class="col-owner">${ownerEditorHtml(task)}</td>
        <td class="col-pl">${taskPlChipsHtml(task)}</td>
        <td class="col-group">${selectHtml("group_id", state.data.groups.map((g) => [g.id, g.title]), task.group_id)}</td>
        <td class="col-special">${selectHtml("special_id", [["","普通事项"], ...state.data.specials.map((s) => [s.id, s.title])], task.special_id || "")}</td>
        <td class="col-date">
          <div class="date-edit-row"><input type="date" data-field="start_date" value="${escapeAttr(task.start_date)}"> ~ <input type="date" data-field="end_date" value="${escapeAttr(task.end_date)}"></div>
          ${taskDdlCountdownHtml(task)}
        </td>
        <td class="col-commit"><input type="date" data-field="recommit_date" value="${escapeAttr(task.recommit_date || "")}"></td>
        <td class="col-done-date"><input type="date" data-field="done_date" value="${escapeAttr(task.done_date || "")}"></td>
        <td class="col-pr-required">${selectHtml("pr_required", [[1, "需 PR"], [0, "仅报告"]], normalizeBooleanFlag(task.pr_required, true))}</td>
        <td class="col-pr">${prLinkEditorHtml(task)}</td>
        <td class="col-report"><input class="link-input" data-field="test_report" placeholder="报告 URL" value="${escapeAttr(task.test_report || "")}"></td>
        <td>${selectHtml("status", STATUS_OPTIONS, task.status)}</td>
        <td class="edit-only">${taskActionButtonsHtml(task)}</td>
      </tr>
    `;
  }).join("");
  document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => handleTaskAction(button).catch(showError)));
  document.querySelectorAll("#rows [data-field]").forEach((control) => control.addEventListener("change", () => markTaskDirty(control)));
  document.querySelectorAll('#rows [data-field="owner"]').forEach((control) => control.addEventListener("change", () => syncOwnerPickerFromInput(control)));
  document.querySelectorAll("#rows [data-owner-picker-toggle]").forEach((control) => control.addEventListener("click", () => toggleOwnerPicker(control)));
  document.querySelectorAll("#rows [data-owner-picker-value]").forEach((control) => control.addEventListener("change", () => syncOwnerFromPicker(control)));
  document.querySelectorAll("#rows [data-operator-picker-toggle]").forEach((control) => control.addEventListener("click", () => toggleOperatorPicker(control)));
  document.querySelectorAll("#rows [data-operator-picker-value]").forEach((control) => control.addEventListener("change", () => syncOperatorFromPicker(control)));
  document.querySelectorAll("#rows [data-pr-append]").forEach((control) => control.addEventListener("click", () => appendPrFromSearch(control)));
  document.querySelectorAll("#rows [data-pr-search]").forEach((control) => control.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closePrSuggestMenus();
      return;
    }
    if (event.key !== "Enter") return;
    event.preventDefault();
    appendPrFromSearch(control);
    closePrSuggestMenus();
  }));
  document.querySelectorAll("#rows [data-pr-search]").forEach((control) => {
    control.addEventListener("focus", () => renderPrSuggestions(control, true));
    control.addEventListener("input", () => renderPrSuggestions(control, true));
  });
  document.querySelectorAll("#rows [data-pr-suggest-menu]").forEach((menu) => menu.addEventListener("mousedown", (event) => {
    const button = event.target.closest("[data-pr-suggest]");
    if (!button) return;
    event.preventDefault();
    choosePrSuggestion(button);
  }));
}

function attachGanttDrag(bar) {
  if (bar.dataset.autoExtended === "true") return;
  bar.addEventListener("mousemove", (event) => {
    const task = state.data.tasks.find((item) => item.id === bar.dataset.taskId);
    if (!canScheduleTask(task) || bar.classList.contains("dragging")) return;
    bar.style.cursor = event.offsetX <= 8 || bar.offsetWidth - event.offsetX <= 8 ? "ew-resize" : "grab";
  });
  bar.addEventListener("pointerdown", (event) => {
    const task = state.data.tasks.find((item) => item.id === bar.dataset.taskId);
    if (!canScheduleTask(task)) return;
    event.preventDefault();
    rememberTaskBaseline(task.id);
    const segments = task.segments?.length ? task.segments : [{ id: `seg-${crypto.randomUUID().slice(0, 10)}`, start_date: task.start_date, end_date: task.end_date, reason: "", position: 0 }];
    task.segments = segments;
    const segment = segments[Number(bar.dataset.segmentIndex || 0)];
    if (!segment) return;
    const track = bar.parentElement;
    const rect = track.getBoundingClientRect();
    const unit = rect.width / state.timeline.total;
    const startX = event.clientX;
    const edge = event.offsetX <= 8 ? "start" : (bar.offsetWidth - event.offsetX <= 8 ? "end" : "move");
    const original = { start: segment.start_date, end: segment.end_date };
    const originalStartOffset = clamp(daysBetween(state.timeline.start, segment.start_date), 0, state.timeline.total - 1);
    const originalEndOffset = clamp(daysBetween(state.timeline.start, segment.end_date), originalStartOffset, state.timeline.total - 1);
    const originalSpan = originalEndOffset - originalStartOffset + 1;
    bar.setPointerCapture(event.pointerId);
    bar.classList.add("dragging");

    const move = (moveEvent) => {
      const deltaDays = Math.round((moveEvent.clientX - startX) / unit);
      let nextStart = originalStartOffset;
      let nextEnd = originalEndOffset;
      if (edge === "move") {
        nextStart = clamp(originalStartOffset + deltaDays, 0, Math.max(0, state.timeline.total - originalSpan));
        nextEnd = nextStart + originalSpan - 1;
      } else if (edge === "start") {
        nextStart = clamp(originalStartOffset + deltaDays, 0, originalEndOffset);
      } else {
        nextEnd = clamp(originalEndOffset + deltaDays, originalStartOffset, state.timeline.total - 1);
      }
      segment.start_date = addDays(state.timeline.start, nextStart);
      segment.end_date = addDays(state.timeline.start, nextEnd);
      paintBar(bar, nextStart, nextEnd);
    };

    const finish = () => {
      bar.removeEventListener("pointermove", move);
      bar.removeEventListener("pointerup", finish);
      bar.removeEventListener("pointercancel", cancel);
      bar.classList.remove("dragging");
      if (segment.start_date === original.start && segment.end_date === original.end) return;
      syncTaskDatesFromSegments(task);
      markTaskDirtyById(task.id);
      render();
    };

    const cancel = () => {
      segment.start_date = original.start;
      segment.end_date = original.end;
      bar.classList.remove("dragging");
      render();
    };

    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", finish);
    bar.addEventListener("pointercancel", cancel);
  });
}

function paintBar(bar, startOffset, endOffset) {
  const width = Math.max(1.2, (endOffset - startOffset + 1) / state.timeline.total * 100);
  const left = startOffset / state.timeline.total * 100;
  bar.style.left = `${left}%`;
  bar.style.width = `${width}%`;
}

function renderAdmin() {
  if (!isAdminEditMode()) {
    $("#groupAdmin").innerHTML = "";
    $("#specialAdmin").innerHTML = "";
    $("#operatorAdmin").innerHTML = "";
    $("#personAdmin").innerHTML = "";
    return;
  }
  $("#groupAdmin").innerHTML = state.data.groups.map((group) => `
    <div class="admin-item" data-group-id="${escapeAttr(group.id)}">
      <div><strong>${escapeHtml(group.title)}</strong><small>${escapeHtml(group.start_date)} ~ ${escapeHtml(group.end_date)}</small></div>
      <span class="ops"><button data-admin="edit-group">编辑</button><button class="danger" data-admin="delete-group">删除</button></span>
    </div>
  `).join("");
  $("#specialAdmin").innerHTML = state.data.specials.map((special) => `
    <div class="admin-item" data-special-id="${escapeAttr(special.id)}">
      <div><strong>${escapeHtml(special.title)}</strong><small>${escapeHtml(groupTitle(special.group_id))} · ${taskCountForSpecial(special.id)} 项</small></div>
      <span class="ops"><button data-admin="edit-special">编辑</button><button class="danger" data-admin="delete-special">删除</button></span>
    </div>
  `).join("");
  const operators = (state.data.operators || []).map(normalizeOperator);
  $("#operatorAdmin").innerHTML = operators.length ? operators.map((operator) => `
    <div class="admin-item" data-operator-id="${escapeAttr(operator.id)}">
      <div>
        <strong>${escapeHtml(operator.label)}</strong>
        <small>别名：${escapeHtml(operator.aliases.join("、") || "-")}</small>
        <small>责任人规则：${escapeHtml(formatOperatorOwnerRules(operator.owner_rules))}</small>
      </div>
      <span class="ops"><button data-admin="edit-operator">编辑</button><button class="danger" data-admin="delete-operator">删除</button></span>
    </div>
  `).join("") : `<p class="empty">暂无算子。只有管理员可以新增算子。</p>`;
  ensurePeopleCatalog();
  const editablePeople = state.data.people.filter((person) => !person.placeholder && !isPlaceholderOwner(person.name));
  $("#personAdmin").innerHTML = editablePeople.length ? editablePeople.map((person) => {
    const assignmentCount = tasksForPerson(person).length;
    const idle = personIsIdleInView(person, filteredTasks().filter(taskIntersectsView));
    return `
      <div class="admin-item ${state.dirtyPersonIds.has(person.id) ? "dirty" : ""}" data-person-id="${escapeAttr(person.id)}">
        <div>
          <strong>${personChipHtml(person)}</strong>
          <label class="admin-pl-label">PL ${plSelectHtml(person.pl)}</label>
          <small>${assignmentCount} 项关联任务 · ${idle ? "当前窗口空闲" : "当前窗口有排期"}</small>
        </div>
        <span class="ops">
          <button data-admin="edit-person">编辑</button>
          <button class="danger" data-admin="delete-person">删除</button>
        </span>
      </div>
    `;
  }).join("") : `<p class="empty">暂无真实人员。系统占位人力不会在这里展示。</p>`;
  document.querySelectorAll("[data-admin]").forEach((button) => button.addEventListener("click", () => handleAdminAction(button).catch(showError)));
  document.querySelectorAll("[data-person-pl]").forEach((select) => select.addEventListener("change", () => handlePersonPlChange(select)));
}

function renderAudit() {
  const recent = state.audit.slice(-10).reverse();
  $("#audit").innerHTML = recent.length
    ? recent.map((item) => `
      <div class="audit-item">
        <time>${escapeHtml(item.ts)}</time>
        <div>
          <strong>${escapeHtml(item.summary || item.action)}</strong>
          ${auditDetailHtml(item)}
        </div>
      </div>
    `).join("")
    : `<p class="empty">暂无变更日志。</p>`;
}

async function handleTaskAction(button) {
  const row = button.closest("tr");
  const taskId = row.dataset.taskId;
  const task = state.data.tasks.find((item) => item.id === taskId);
  if (!task) return;
  const action = button.dataset.action;
  if (!canEditTask(task)) {
    alert("当前账号只能给自己负责的任务提交 PR/转测报告链接；同算子关联任务仅供查看。");
    return;
  }
  if (action === "save") {
    await saveTaskRows([row]);
  }
  if (action === "split") {
    if (!canScheduleTask(task)) {
      alert("普通开发账号不能调整排期。");
      return;
    }
    await splitTask(task.id);
  }
  if (action === "delete") {
    if (!isAdminEditMode()) {
      alert("普通开发账号不能删除任务。");
      return;
    }
    if (!confirmHighRisk("确认删除任务", `任务：${task.title}\n会删除该任务及其甘特分段记录，页面内不能撤销。`)) return;
    if (!(await ensureNoVersionConflict())) return;
    state.data.tasks = state.data.tasks.filter((item) => item.id !== task.id);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("task.delete", "task", task.id, `删除任务：${task.title}`, { title: task.title });
      const result = await workerDelete(`/api/tasks/${encodeURIComponent(task.id)}`, { auditEntry: entry });
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`删除任务：${task.title}`, "task.delete", "task", task.id, { title: task.title });
  }
}

function markTaskDirty(control) {
  const row = control.closest("tr");
  if (!row?.dataset.taskId) return;
  rememberTaskBaseline(row.dataset.taskId);
  applyRowToTask(row, false);
  row.classList.add("dirty");
  state.dirtyTaskIds.add(row.dataset.taskId);
  updateEditStatus();
}

function markTaskDirtyById(taskId) {
  rememberTaskBaseline(taskId);
  state.dirtyTaskIds.add(taskId);
  updateEditStatus();
}

function applyRowToTask(row, normalizeSegments = true) {
  const task = state.data.tasks.find((item) => item.id === row.dataset.taskId);
  if (!task) return null;
  row.querySelectorAll("[data-field]").forEach((input) => {
    task[input.dataset.field] = input.dataset.field === "operator_ids" ? normalizeOperatorIdsText(input.value) : input.value.trim();
  });
  const changed = syncTaskDeliveryRules(task);
  if (changed.includes("risk")) {
    const riskControl = row.querySelector('[data-field="risk"]');
    if (riskControl) riskControl.value = task.risk;
  }
  if (changed.includes("status")) {
    const statusControl = row.querySelector('[data-field="status"]');
    if (statusControl) statusControl.value = task.status;
  }
  task.updated_at = nowIso();
  if (normalizeSegments && (!task.segments?.length || task.segments.length <= 1)) normalizeTaskSegments(task);
  return task;
}

function syncTaskDatesFromSegments(task) {
  const segments = [...(task.segments || [])].sort((a, b) => a.start_date.localeCompare(b.start_date));
  if (!segments.length) return;
  task.segments = segments.map((segment, index) => ({ ...segment, position: index }));
  task.start_date = segments[0].start_date;
  task.end_date = segments[segments.length - 1].end_date;
  task.updated_at = nowIso();
}

function confirmHighRisk(title, detail) {
  return confirm(`${title}\n\n${detail}\n\n这是高风险操作，请确认继续。`);
}

function changeTouchesSchedule(change) {
  return Object.keys(change?.changes || {}).some((field) => SCHEDULE_CHANGE_FIELDS.has(field));
}

function promptScheduleChangeReason(task, change) {
  if (!changeTouchesSchedule(change)) return true;
  const reason = prompt(`任务「${task.title}」排期发生变化，请填写变更原因（必填）：`, String(task.notes || "").trim());
  if (!reason || !reason.trim()) {
    alert("改任务排期必须填写变更原因，本次保存已取消。");
    return false;
  }
  const cleanReason = reason.trim();
  task.notes = cleanReason;
  if (Array.isArray(task.segments)) {
    task.segments = task.segments.map((segment) => ({
      ...segment,
      reason: String(segment.reason || "").trim() || cleanReason,
    }));
  }
  task.updated_at = nowIso();
  const refreshed = taskAuditChange(task);
  if (refreshed) change.changes = refreshed.changes;
  return true;
}

async function ensureNoVersionConflict() {
  if (!WORKER_API_BASE || !state.serverVersion) return true;
  const result = await workerGet("/api/version");
  const latest = result.version || "";
  if (!latest || latest === state.serverVersion) return true;
  state.pendingRemoteVersion = latest;
  updateEditStatus();
  const shouldRefresh = confirm(
    "后台数据已被其他人更新。为避免覆盖他人修改，本次保存已暂停。\n\n" +
    "点击“确定”刷新页面数据（会放弃当前未保存修改）；点击“取消”保留当前页面，手动复制修改后再刷新合并。"
  );
  if (shouldRefresh) {
    state.dirtyTaskIds.clear();
    state.dirtyPersonIds.clear();
    state.taskBaselines.clear();
    state.personBaselines.clear();
    state.serverVersion = latest;
    await load();
  }
  return false;
}

async function saveAllTasks() {
  if (!state.dirtyTaskIds.size && !state.dirtyPersonIds.size) {
    alert("没有待保存的变更。");
    return;
  }
  if (state.dirtyTaskIds.size) {
    const savedTasks = await saveTaskRows();
    if (savedTasks === false) return;
  }
  if (state.dirtyPersonIds.size) await saveDirtyPeople();
}

async function saveTaskRows(extraRows = []) {
  const rows = new Map([...document.querySelectorAll("#rows tr.dirty"), ...extraRows]
    .filter((row) => row?.dataset?.taskId)
    .map((row) => [row.dataset.taskId, row]));
  extraRows.forEach((row) => {
    if (!row?.dataset?.taskId) return;
    rememberTaskBaseline(row.dataset.taskId);
    state.dirtyTaskIds.add(row.dataset.taskId);
  });
  const ids = [...state.dirtyTaskIds];
  if (!ids.length) {
    alert("没有待保存的任务变更。");
    return false;
  }
  if (!(await ensureNoVersionConflict())) return false;
  ids.forEach((id) => {
    const row = rows.get(id);
    if (row) applyRowToTask(row);
  });
  const changes = ids.map((id) => taskAuditChange(state.data.tasks.find((task) => task.id === id))).filter(Boolean);
  if (!changes.length) {
    state.dirtyTaskIds.clear();
    state.taskBaselines.clear();
    render();
    alert("没有检测到字段变更。");
    return false;
  }
  for (const change of changes) {
    const task = state.data.tasks.find((item) => item.id === change.id);
    if (task && !promptScheduleChangeReason(task, change)) return false;
  }
  const summary = changes.length === 1 ? `更新任务：${changes[0].title}` : `批量更新任务：${changes.length}项`;
  const action = changes.length === 1 ? "task.update" : "task.batch_update";
  const id = changes.length === 1 ? changes[0].id : "batch";
  if (WORKER_API_BASE) {
    $("#editStatus").textContent = "正在按字段写入 Cloudflare D1...";
    for (const change of changes) {
      const task = state.data.tasks.find((item) => item.id === change.id);
      if (!task) continue;
      const entry = {
        ts: nowIso(),
        action: "task.patch",
        entity: "task",
        id: task.id,
        summary: `更新任务：${change.title}`,
        detail: { ids: [task.id], changes: [change] },
        source: "cloudflare-d1",
      };
      const fields = taskPatchFieldsFromChange(task, change);
      if (!Object.keys(fields).length) continue;
      const result = await patchTask(task, fields, entry);
      if (result.entry) state.audit.push(result.entry);
    }
    state.dirtyTaskIds.clear();
    state.taskBaselines.clear();
    $("#editStatus").textContent = "已按字段写入 Cloudflare D1";
    render();
    return true;
  }
  await saveRepository(summary, action, "task", id, { ids: changes.map((change) => change.id), changes });
  return true;
}

function taskPatchFieldsFromChange(task, change) {
  const fields = {};
  Object.keys(change.changes || {}).forEach((field) => {
    if (isDeveloperEditMode() && (field === "risk" || field === "status" || field === "done_date" || field === "pr_required")) return;
    fields[field] = field === "segments" ? (task.segments || []) : task[field];
  });
  return fields;
}

async function addTask() {
  const title = prompt("任务名称：", "新任务");
  if (!title) return;
  const firstGroup = state.data.groups[0];
  const today = todayBjYmd();
  const task = {
    id: `task-${crypto.randomUUID().slice(0, 10)}`,
    title,
    scope: "",
    target: "",
    owner: "待排人力",
    status: "todo",
    risk: "中",
    priority: "P1",
    group_id: firstGroup?.id || "",
    special_id: null,
    start_date: today,
    end_date: today,
    evidence: [],
    dependencies: [],
    pr_required: 1,
    pr_link: "",
    test_report: "",
    notes: "",
    recommit_date: "",
    done_date: "",
    operator_ids: "",
    position: state.data.tasks.length,
    created_at: nowIso(),
    updated_at: nowIso(),
    segments: [],
  };
  normalizeTaskSegments(task);
  state.data.tasks.push(task);
  if (WORKER_API_BASE) {
    const entry = cloudflareAuditEntry("task.create", "task", task.id, `新增任务：${title}`, { title });
    const result = await workerPost("/api/tasks", { task, auditEntry: entry });
    if (result.task) mergeTask(result.task);
    if (result.prCatalog) state.prCatalog = result.prCatalog;
    applyCloudflareMutationResult(result);
    if (result.entry) state.audit.push(result.entry);
    render();
    return;
  }
  await saveRepository(`新增任务：${title}`, "task.create", "task", task.id, { title });
}

async function splitTask(taskId) {
  const task = state.data.tasks.find((item) => item.id === taskId);
  if (!task) return;
  rememberTaskBaseline(task.id);
  const breakStart = prompt("中断开始日期 YYYY-MM-DD：", task.start_date);
  if (!breakStart) return;
  const breakEnd = prompt("中断结束日期 YYYY-MM-DD：", breakStart);
  if (!breakEnd) return;
  const reason = prompt("中断原因：", "临时插入其它任务");
  if (!reason) return;
  if (WORKER_API_BASE && !(await ensureNoVersionConflict())) return;
  const segments = task.segments?.length ? task.segments : [{ id: `seg-${crypto.randomUUID().slice(0, 10)}`, start_date: task.start_date, end_date: task.end_date, reason: "", position: 0 }];
  const next = [];
  for (const segment of segments) {
    if (breakEnd < segment.start_date || breakStart > segment.end_date) {
      next.push(segment);
      continue;
    }
    if (breakStart > segment.start_date) next.push({ ...segment, id: `seg-${crypto.randomUUID().slice(0, 10)}`, end_date: prevDay(breakStart) });
    if (breakEnd < segment.end_date) next.push({ ...segment, id: `seg-${crypto.randomUUID().slice(0, 10)}`, start_date: nextDay(breakEnd) });
  }
  if (!next.length) {
    alert("中断时间覆盖了整个任务，无法切分。");
    return;
  }
  task.segments = next.map((segment, index) => ({ ...segment, reason: segment.reason || reason, position: index }));
  task.notes = reason;
  task.updated_at = nowIso();
  syncTaskDatesFromSegments(task);
  if (WORKER_API_BASE) {
    const change = taskAuditChange(task);
    const entry = {
      ts: nowIso(),
      action: "task.split",
      entity: "task",
      id: task.id,
      summary: `切分任务：${task.title}`,
      detail: { break_start: breakStart, break_end: breakEnd, reason, changes: change ? [change] : [] },
      source: "cloudflare-d1",
    };
    const fields = change ? taskPatchFieldsFromChange(task, change) : { segments: task.segments, start_date: task.start_date, end_date: task.end_date, notes: task.notes };
    const result = await patchTask(task, fields, entry);
    if (result.entry) state.audit.push(result.entry);
    state.dirtyTaskIds.delete(task.id);
    state.taskBaselines.delete(task.id);
    render();
    return;
  }
  await saveRepository(`切分任务：${task.title}`, "task.split", "task", task.id, { break_start: breakStart, break_end: breakEnd, reason });
}

async function addGroup() {
  const title = prompt("分组名称：", "新转测分组");
  if (!title) return;
  const due = prompt("截止日期 YYYY-MM-DD：", "2026-06-25");
  if (!due) return;
  const group = { id: `group-${crypto.randomUUID().slice(0, 10)}`, title, due_date: due, start_date: due, end_date: due, position: state.data.groups.length };
  state.data.groups.push(group);
  if (WORKER_API_BASE) {
    const entry = cloudflareAuditEntry("group.create", "group", group.id, `新增分组：${title}`, { title });
    const result = await workerPost("/api/groups", { group, auditEntry: entry });
    if (result.group) mergeEntityItem("groups", result.group);
    applyCloudflareMutationResult(result);
    if (result.entry) state.audit.push(result.entry);
    render();
    return;
  }
  await saveRepository(`新增分组：${title}`, "group.create", "group", group.id, { title });
}

async function addSpecial() {
  const title = prompt("专项名称：", "专项：新专项");
  if (!title) return;
  const special = { id: `special-${crypto.randomUUID().slice(0, 10)}`, title, group_id: state.data.groups[0]?.id || "", position: state.data.specials.length, collapsed: 0 };
  state.data.specials.push(special);
  if (WORKER_API_BASE) {
    const entry = cloudflareAuditEntry("special.create", "special", special.id, `新增专项：${title}`, { title });
    const result = await workerPost("/api/specials", { special, auditEntry: entry });
    if (result.special) mergeEntityItem("specials", result.special);
    applyCloudflareMutationResult(result);
    if (result.entry) state.audit.push(result.entry);
    render();
    return;
  }
  await saveRepository(`新增专项：${title}`, "special.create", "special", special.id, { title });
}

async function addOperator() {
  if (!isAdminEditMode()) return alert("只有管理员可以新增算子。");
  const rawId = prompt("算子 ID / 全名：", "new_operator");
  if (rawId === null) return;
  const id = normalizeOperatorId(rawId);
  if (!id) return alert("算子 ID 不能为空。");
  ensureOperatorCatalog();
  if ((state.data.operators || []).some((operator) => operator.id === id)) return alert("算子已存在。");
  const rawLabel = prompt("页面显示名称：", id);
  if (rawLabel === null) return;
  const label = String(rawLabel || id).trim() || id;
  const rawAliases = prompt("标题别名（逗号/顿号/换行分隔，可为空）：", label === id ? id : `${label},${id}`);
  if (rawAliases === null) return;
  const aliases = normalizeOperatorAliases(rawAliases || label);
  const rawRules = prompt("责任人规则（可空；示例：2026-06-30:张三;*:李四）：", "");
  if (rawRules === null) return;
  const operator = {
    id,
    label,
    aliases,
    owner_rules: parseOperatorOwnerRulesInput(rawRules),
    position: state.data.operators.length,
    active: true,
  };
  state.data.operators.push(operator);
  if (WORKER_API_BASE) {
    const entry = cloudflareAuditEntry("operator.create", "operator", operator.id, `新增算子：${label}`, { id, label });
    const result = await workerPost("/api/operators", { operator, auditEntry: entry });
    if (result.operator) mergeEntityItem("operators", result.operator);
    applyCloudflareMutationResult(result);
    if (result.entry) state.audit.push(result.entry);
    render();
    return;
  }
  await saveRepository(`新增算子：${label}`, "operator.create", "operator", operator.id, { id, label });
}

async function addPerson() {
  const rawName = prompt("人员姓名：", "");
  if (rawName === null) return;
  const name = normalizeOwnerName(rawName);
  if (!name || isPlaceholderOwner(name)) return alert("人员姓名不能为空，也不能使用系统占位名称。");
  ensurePeopleCatalog();
  if ((state.data.people || []).some((person) => person.name === name)) return alert("人员已存在。");
  const rawPl = prompt(`PL（从以下列表中填写）：${PL_OPTIONS.join("、")}`, DEFAULT_PL);
  if (rawPl === null) return;
  const pl = String(rawPl || "").trim();
  if (!PL_OPTIONS.includes(pl)) return alert(`PL 需从列表中选择：${PL_OPTIONS.join("、")}`);
  const rawEmail = prompt("邮箱（仅写入后端账号，不展示在项目看板）：", "");
  if (rawEmail === null) return;
  const email = normalizeEmail(rawEmail);
  if (!isValidEmail(email)) return alert("邮箱不能为空，且需要是有效邮箱格式。");
  if (!WORKER_API_BASE) return alert("新增人员自动创建账号需要 Cloudflare D1 后端模式。");
  const maxPosition = Math.max(-1, ...(state.data.people || []).map((person) => Number(person.position) || 0));
  const person = { id: `person-${crypto.randomUUID().slice(0, 10)}`, name, position: maxPosition + 1, pl, placeholder: false };
  if (WORKER_API_BASE) {
    const entry = cloudflareAuditEntry("person.create", "person", person.id, `新增人员：${name}`, { name, pl });
    const result = await workerPost("/api/people", { person, email, createAccount: true, auditEntry: entry });
    if (result.person) mergeEntityItem("people", result.person);
    applyCloudflareMutationResult(result);
    if (result.entry) state.audit.push(result.entry);
    if (result.account?.password) {
      const message = [
        `${name}，你好：`,
        "",
        "你的 flash-linear-attention-npu 项目看板账号已准备好。",
        "",
        "登录地址：https://fazhenyao.github.io/flash-linear-attention-npu-io/",
        `账号：${result.account.username || name}`,
        `初始密码：${result.account.password}`,
        `绑定邮箱：${result.account.email || email}`,
        "",
        "首次登录后建议点击“修改密码”更新为自己的密码。",
      ].join("\n");
      prompt("人员已新增并创建账号，请复制以下通知内容：", message);
    } else if (result.account?.status === "existing") {
      alert("人员已新增，但同名账号已存在，未重置密码。需要新密码时请在账号管理中重置。");
    }
    render();
    return;
  }
}

function rememberPersonBaseline(personId) {
  if (!personId || state.personBaselines.has(personId)) return;
  const person = state.data.people.find((item) => item.id === personId);
  if (person) state.personBaselines.set(personId, { name: person.name, pl: normalizePl(person.pl) });
}

async function saveDirtyPeople() {
  const ids = [...state.dirtyPersonIds];
  if (!ids.length) return true;
  if (!(await ensureNoVersionConflict())) return false;
  const changes = ids.map((id) => {
    const person = state.data.people.find((item) => item.id === id);
    const before = state.personBaselines.get(id);
    if (!person || !before) return null;
    const nextPl = normalizePl(person.pl);
    return before.pl === nextPl ? null : { id, name: person.name, from: before.pl, to: nextPl };
  }).filter(Boolean);
  if (!changes.length) {
    state.dirtyPersonIds.clear();
    state.personBaselines.clear();
    render();
    return false;
  }
  if (WORKER_API_BASE) {
    $("#editStatus").textContent = "正在写入人员 PL 到 Cloudflare D1...";
    for (const change of changes) {
      const entry = cloudflareAuditEntry("person.patch", "person", change.id, `更新人员 PL：${change.name} -> ${change.to}`, {
        name: change.name,
        pl: change.to,
        from: change.from,
      });
      const result = await workerPatch(`/api/people/${encodeURIComponent(change.id)}`, {
        fields: { pl: change.to },
        auditEntry: entry,
      });
      if (result.person) mergeEntityItem("people", result.person);
      syncCloudflareVersion(result);
      if (result.entry) state.audit.push(result.entry);
    }
    state.dirtyPersonIds.clear();
    state.personBaselines.clear();
    $("#editStatus").textContent = "已写入人员 PL 到 Cloudflare D1";
    render();
    return true;
  }
  const summary = changes.length === 1 ? `更新人员 PL：${changes[0].name} -> ${changes[0].to}` : `批量更新人员 PL：${changes.length}项`;
  await saveRepository(summary, changes.length === 1 ? "person.update_pl" : "person.batch_update_pl", changes.length === 1 ? "person" : "people", changes.length === 1 ? changes[0].id : "batch", { changes });
  state.dirtyPersonIds.clear();
  state.personBaselines.clear();
  render();
  return true;
}

function handlePersonPlChange(select) {
  const personId = select.closest("[data-person-id]")?.dataset.personId;
  const person = state.data.people.find((item) => item.id === personId);
  if (!person || person.placeholder) return;
  const pl = normalizePl(select.value);
  if (normalizePl(person.pl) === pl) return;
  rememberPersonBaseline(person.id);
  person.pl = pl;
  const baseline = state.personBaselines.get(person.id);
  if (baseline && baseline.pl === pl) {
    state.dirtyPersonIds.delete(person.id);
    state.personBaselines.delete(person.id);
  } else {
    state.dirtyPersonIds.add(person.id);
  }
  render();
}

async function handleAdminAction(button) {
  if (!isAdminEditMode()) {
    alert("当前账号没有管理员权限。");
    return;
  }
  const action = button.dataset.admin;
  const groupId = button.closest("[data-group-id]")?.dataset.groupId;
  const specialId = button.closest("[data-special-id]")?.dataset.specialId;
  const operatorId = button.closest("[data-operator-id]")?.dataset.operatorId;
  const personId = button.closest("[data-person-id]")?.dataset.personId;
  if (action === "edit-group") {
    const group = state.data.groups.find((item) => item.id === groupId);
    const title = prompt("分组名称：", group.title);
    if (!title) return;
    const due = prompt("截止日期 YYYY-MM-DD：", group.due_date);
    if (!due) return;
    Object.assign(group, { title, due_date: due, end_date: due });
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("group.patch", "group", group.id, `更新分组：${title}`, { title, due_date: due });
      const result = await workerPatch(`/api/groups/${encodeURIComponent(group.id)}`, {
        fields: { title, due_date: due, end_date: due },
        auditEntry: entry,
      });
      if (result.group) mergeEntityItem("groups", result.group);
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`更新分组：${title}`, "group.update", "group", group.id, { title });
  }
  if (action === "delete-group") {
    const group = state.data.groups.find((item) => item.id === groupId);
    if (!group || state.data.groups.length <= 1) return alert("至少保留一个分组。");
    const fallback = state.data.groups.find((item) => item.id !== group.id);
    if (!confirmHighRisk("确认删除分组", `分组：${group.title}\n不会删除任何任务；关联任务和专项的分组字段会转入默认分组：${fallback.title}。`)) return;
    if (!(await ensureNoVersionConflict())) return;
    state.data.tasks.forEach((task) => { if (task.group_id === group.id) task.group_id = fallback.id; });
    state.data.specials.forEach((special) => { if (special.group_id === group.id) special.group_id = fallback.id; });
    state.data.groups = state.data.groups.filter((item) => item.id !== group.id);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("group.delete", "group", group.id, `删除分组：${group.title}`, { fallback_group_id: fallback.id });
      const result = await workerDelete(`/api/groups/${encodeURIComponent(group.id)}`, {
        fallback_group_id: fallback.id,
        auditEntry: entry,
      });
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`删除分组：${group.title}`, "group.delete", "group", group.id, { fallback_group_id: fallback.id });
  }
  if (action === "edit-special") {
    const special = state.data.specials.find((item) => item.id === specialId);
    const title = prompt("专项名称：", special.title);
    if (!title) return;
    special.title = title;
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("special.patch", "special", special.id, `更新专项：${title}`, { title });
      const result = await workerPatch(`/api/specials/${encodeURIComponent(special.id)}`, {
        fields: { title },
        auditEntry: entry,
      });
      if (result.special) mergeEntityItem("specials", result.special);
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`更新专项：${title}`, "special.update", "special", special.id, { title });
  }
  if (action === "delete-special") {
    const special = state.data.specials.find((item) => item.id === specialId);
    if (!special) return;
    if (!confirmHighRisk("确认删除专项", `专项：${special.title}\n不会删除任何任务；关联任务的专项字段会变为默认字段：普通事项。`)) return;
    if (!(await ensureNoVersionConflict())) return;
    state.data.tasks.forEach((task) => { if (task.special_id === special.id) task.special_id = null; });
    state.data.specials = state.data.specials.filter((item) => item.id !== special.id);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("special.delete", "special", special.id, `删除专项：${special.title}`, { title: special.title });
      const result = await workerDelete(`/api/specials/${encodeURIComponent(special.id)}`, { auditEntry: entry });
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`删除专项：${special.title}`, "special.delete", "special", special.id, { title: special.title });
  }
  if (action === "edit-operator") {
    const operator = state.data.operators.find((item) => item.id === operatorId);
    if (!operator) return;
    const label = prompt("算子显示名称：", operator.label || operator.id);
    if (!label) return;
    const aliasesText = prompt("标题别名（逗号/顿号/换行分隔）：", normalizeOperatorAliases(operator.aliases).join(","));
    if (aliasesText === null) return;
    const ownerRulesText = prompt("责任人规则（可空；示例：2026-06-30:张三;*:李四）：", operatorOwnerRulesInputText(operator.owner_rules));
    if (ownerRulesText === null) return;
    const fields = {
      label: label.trim(),
      aliases: normalizeOperatorAliases(aliasesText),
      owner_rules: parseOperatorOwnerRulesInput(ownerRulesText),
      active: true,
    };
    Object.assign(operator, fields);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("operator.patch", "operator", operator.id, `更新算子：${fields.label}`, { id: operator.id, label: fields.label });
      const result = await workerPatch(`/api/operators/${encodeURIComponent(operator.id)}`, { fields, auditEntry: entry });
      if (result.operator) mergeEntityItem("operators", result.operator);
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`更新算子：${fields.label}`, "operator.update", "operator", operator.id, { id: operator.id });
  }
  if (action === "delete-operator") {
    const operator = state.data.operators.find((item) => item.id === operatorId);
    if (!operator) return;
    const linkedCount = state.data.tasks.filter((task) => parseOperatorIds(task.operator_ids).includes(operator.id)).length;
    if (!confirmHighRisk("确认删除算子", `算子：${operator.label || operator.id}\n不会删除任何任务；${linkedCount} 个显式关联任务会移除此算子字段。`)) return;
    if (!(await ensureNoVersionConflict())) return;
    state.data.tasks.forEach((task) => {
      const next = parseOperatorIds(task.operator_ids).filter((id) => id !== operator.id).join("/");
      if (next !== String(task.operator_ids || "")) task.operator_ids = next;
    });
    state.data.operators = state.data.operators.filter((item) => item.id !== operator.id);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("operator.delete", "operator", operator.id, `删除算子：${operator.label || operator.id}`, { id: operator.id });
      const result = await workerDelete(`/api/operators/${encodeURIComponent(operator.id)}`, { auditEntry: entry });
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`删除算子：${operator.label || operator.id}`, "operator.delete", "operator", operator.id, { id: operator.id });
  }
  if (action === "edit-person") {
    const person = state.data.people.find((item) => item.id === personId);
    if (!person || person.placeholder) return;
    const name = normalizeOwnerName(prompt("人员姓名：", person.name));
    if (!name || isPlaceholderOwner(name)) return alert("人员姓名不能为空，也不能使用系统占位名称。");
    if (name !== person.name && state.data.people.some((item) => item.name === name)) return alert("人员已存在。");
    const oldName = person.name;
    person.name = name;
    person.placeholder = false;
    renameOwnerInTasks(oldName, name);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("person.patch", "person", person.id, `更新人员：${oldName} -> ${name}`, { old_name: oldName, name });
      const result = await workerPatch(`/api/people/${encodeURIComponent(person.id)}`, {
        fields: { name, placeholder: false },
        auditEntry: entry,
      });
      if (result.person) mergeEntityItem("people", result.person);
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`更新人员：${oldName} -> ${name}`, "person.update", "person", person.id, { old_name: oldName, name });
  }
  if (action === "delete-person") {
    const person = state.data.people.find((item) => item.id === personId);
    if (!person || person.placeholder) return;
    const assignments = tasksForPerson(person);
    if (assignments.length) return alert(`该人员仍关联 ${assignments.length} 项任务，请先调整任务责任人。`);
    if (!confirmHighRisk("确认删除人员", `人员：${person.name}\n仅删除人员档案；不会删除任务。已确认该人员当前没有关联任务。`)) return;
    if (!(await ensureNoVersionConflict())) return;
    state.data.people = state.data.people.filter((item) => item.id !== person.id);
    if (WORKER_API_BASE) {
      const entry = cloudflareAuditEntry("person.delete", "person", person.id, `删除人员：${person.name}`, { name: person.name });
      const result = await workerDelete(`/api/people/${encodeURIComponent(person.id)}`, { auditEntry: entry });
      applyCloudflareMutationResult(result);
      if (result.entry) state.audit.push(result.entry);
      render();
      return;
    }
    await saveRepository(`删除人员：${person.name}`, "person.delete", "person", person.id, { name: person.name });
  }
}

function renameOwnerInTasks(oldName, newName) {
  (state.data.tasks || []).forEach((task) => {
    if (!task.owner) return;
    const parts = String(task.owner).split(/([、/,，;；&]+)/);
    const next = parts.map((part) => normalizeOwnerName(part) === oldName ? newName : part).join("");
    if (next !== task.owner) {
      task.owner = next;
      task.updated_at = nowIso();
    }
  });
}

async function patchTask(task, fields, auditEntry) {
  requireToken();
  const result = await workerPatch(`/api/tasks/${encodeURIComponent(task.id)}`, {
    fields,
    auditEntry,
    expectedVersion: state.serverVersion || "",
  });
  if (result.task) mergeTask(result.task);
  if (result.prCatalog) state.prCatalog = result.prCatalog;
  if (result.version) {
    state.serverVersion = result.version;
    state.pendingRemoteVersion = "";
    if (state.data) state.data.version = result.version;
  }
  return result;
}

function mergeTask(nextTask) {
  const index = (state.data.tasks || []).findIndex((task) => task.id === nextTask.id);
  if (index >= 0) state.data.tasks[index] = nextTask;
}

function mergeEntityItem(collection, nextItem) {
  const list = state.data[collection] || [];
  const index = list.findIndex((item) => item.id === nextItem.id);
  if (index >= 0) list[index] = nextItem;
  else list.push(nextItem);
  state.data[collection] = list;
}

function cloudflareAuditEntry(action, entity, id, summary, detail = {}) {
  return { ts: nowIso(), action, entity, id, summary, detail, source: "cloudflare-d1" };
}

function syncCloudflareVersion(result) {
  if (result.version) {
    state.serverVersion = result.version;
    state.pendingRemoteVersion = "";
    if (state.data) state.data.version = result.version;
  }
}

function applyCloudflareMutationResult(result) {
  syncCloudflareVersion(result);
  state.dirtyTaskIds.clear();
  state.taskBaselines.clear();
}

async function saveRepository(summary, action, entity, id, detail = {}) {
  requireToken();
  ensureOperatorCatalog();
  ensurePeopleCatalog();
  syncAllTaskDeliveryRules();
  state.data.generatedAt = nowIso();
  const entry = { ts: nowIso(), action, entity, id, summary, detail, source: WORKER_API_BASE ? "cloudflare-d1" : "github-pages" };
  if (WORKER_API_BASE) {
    $("#editStatus").textContent = "正在写入 Cloudflare D1...";
    const result = await workerPost("/api/save", {
      state: state.data,
      auditEntry: entry,
      prCatalog: state.prCatalog,
      expectedVersion: state.serverVersion || "",
    });
    if (result.state) state.data = result.state;
    if (result.prCatalog) state.prCatalog = result.prCatalog;
    if (result.version) state.serverVersion = result.version;
    state.pendingRemoteVersion = "";
    state.audit.push(entry);
    state.dirtyTaskIds.clear();
    state.dirtyPersonIds.clear();
    state.taskBaselines.clear();
    state.personBaselines.clear();
    $("#editStatus").textContent = "已写入 Cloudflare D1";
    render();
    return;
  }
  state.audit.push(entry);
  const stateText = JSON.stringify(state.data, null, 2) + "\n";
  const auditText = state.audit.map((item) => JSON.stringify(item)).join("\n") + "\n";
  $("#editStatus").textContent = "正在写入 GitHub 仓库...";
  await commitFiles({
    [DATA_PATHS.state]: stateText,
    [DATA_PATHS.pageState]: stateText,
    [DATA_PATHS.audit]: auditText,
    [DATA_PATHS.pageAudit]: auditText,
  }, `记录数据变更: ${summary}`);
  state.dirtyTaskIds.clear();
  state.dirtyPersonIds.clear();
  state.taskBaselines.clear();
  state.personBaselines.clear();
  $("#editStatus").textContent = "已写入 GitHub 仓库，Pages 稍后刷新";
  render();
}

function updateEditStatus() {
  if (!state.token) {
    $("#editStatus").textContent = WORKER_API_BASE ? "只读模式：请登录账号" : "只读模式";
    return;
  }
  const dirtyCount = state.dirtyTaskIds.size + state.dirtyPersonIds.size;
  const target = WORKER_API_BASE ? "Cloudflare D1" : "GitHub 仓库";
  const user = WORKER_API_BASE && state.authUser ? `（${state.authUser.displayName || state.authUser.username} / ${state.authUser.role}）` : "";
  const scope = isDeveloperEditMode() ? "仅显示自己任务及同算子关联任务；只能提交自己任务的 PR/转测报告链接" : "甘特条可拖动，边缘可拉伸";
  if (dirtyCount && state.pendingRemoteVersion) {
    $("#editStatus").textContent = `编辑模式${user}：${dirtyCount} 项待保存；后台已有新数据，请先刷新或手动合并`;
    return;
  }
  $("#editStatus").textContent = dirtyCount ? `编辑模式${user}：${dirtyCount} 项待保存` : `编辑模式${user}：${scope}；保存会写入 ${target}`;
}

function startRealtimeSync() {
  if (!WORKER_API_BASE || state.realtimeTimer) return;
  state.realtimeTimer = window.setInterval(() => {
    checkRemoteVersion().catch(() => {});
  }, 3000);
}

async function checkRemoteVersion() {
  if (!WORKER_API_BASE || !state.data || state.loading) return;
  const result = await workerGet("/api/version");
  const version = result.version || "";
  if (!version || !state.serverVersion) {
    state.serverVersion = version;
    return;
  }
  if (version === state.serverVersion) return;
  if (state.dirtyTaskIds.size || state.dirtyPersonIds.size) {
    state.pendingRemoteVersion = version;
    updateEditStatus();
    return;
  }
  state.serverVersion = version;
  await load();
}

async function commitFiles(files, message) {
  const ref = await gh(`/git/ref/heads/${REPO.branch}`);
  const commit = await gh(`/git/commits/${ref.object.sha}`);
  const treeItems = [];
  for (const [path, content] of Object.entries(files)) {
    const blob = await gh("/git/blobs", {
      method: "POST",
      body: { content, encoding: "utf-8" },
    });
    treeItems.push({ path, mode: "100644", type: "blob", sha: blob.sha });
  }
  const tree = await gh("/git/trees", {
    method: "POST",
    body: { base_tree: commit.tree.sha, tree: treeItems },
  });
  const nextCommit = await gh("/git/commits", {
    method: "POST",
    body: { message, tree: tree.sha, parents: [ref.object.sha] },
  });
  await gh(`/git/refs/heads/${REPO.branch}`, {
    method: "PATCH",
    body: { sha: nextCommit.sha },
  });
}

async function gh(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    method: options.method || "GET",
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${state.token}`,
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.message || `GitHub API ${response.status}`);
  return data;
}

async function workerGet(path) {
  return workerFetch(path);
}

async function workerPost(path, body) {
  return workerFetch(path, {
    method: "POST",
    body,
  });
}

async function workerPatch(path, body) {
  return workerFetch(path, {
    method: "PATCH",
    body,
  });
}

async function workerDelete(path, body = {}) {
  return workerFetch(path, {
    method: "DELETE",
    body,
  });
}

async function workerFetch(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`${WORKER_API_BASE}${path}`, {
    method: options.method || "GET",
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 409 && data.version) {
      state.pendingRemoteVersion = data.version;
      updateEditStatus();
    }
    throw new Error(data.error || `Cloudflare API ${response.status}`);
  }
  return data;
}

function requireToken() {
  if (!state.token) throw new Error(WORKER_API_BASE ? "请先启用编辑模式并输入 Cloudflare ADMIN_TOKEN。" : "请先启用编辑模式并输入 GitHub token。");
}

function enableEditMode() {
  const token = $("#token").value.trim();
  if (!token || token === "********") return alert(WORKER_API_BASE ? "请输入 Cloudflare ADMIN_TOKEN。" : "请输入 GitHub fine-grained token。");
  state.token = token;
  sessionStorage.setItem("flashPagesToken", token);
  render();
}

async function loginWorker() {
  const username = $("#loginUser").value.trim();
  const password = $("#loginPassword").value;
  if (!username || !password) return alert("请输入账号和密码。");
  const data = await workerPost("/api/login", { username, password });
  state.token = data.token;
  state.authUser = data.user;
  sessionStorage.setItem("flashWorkerAuthToken", data.token);
  $("#loginPassword").value = "";
  await load();
}

async function changePassword() {
  if (!WORKER_API_BASE || !state.token) return alert("请先登录账号。");
  const oldPassword = prompt("请输入当前密码：", "");
  if (!oldPassword) return;
  const newPassword = prompt("请输入新密码（至少 8 位）：", "");
  if (!newPassword) return;
  if (newPassword.length < 8) return alert("新密码至少 8 位。");
  const repeatPassword = prompt("请再次输入新密码：", "");
  if (newPassword !== repeatPassword) return alert("两次输入的新密码不一致。");
  if (!confirmHighRisk("确认修改账号密码", "确认后新密码立即生效，旧密码将无法继续登录。")) return;
  await workerPost("/api/me/password", { oldPassword, newPassword });
  alert("密码已修改，请妥善保存新密码。");
}

function logout() {
  state.token = "";
  state.authUser = null;
  sessionStorage.removeItem("flashPagesToken");
  sessionStorage.removeItem("flashWorkerAuthToken");
  render();
}

function normalizeTaskSegments(task) {
  task.segments = [{ id: task.segments?.[0]?.id || `seg-${crypto.randomUUID().slice(0, 10)}`, start_date: task.start_date, end_date: task.end_date, reason: task.notes || "", position: 0 }];
}

function syncAllTaskDeliveryRules() {
  (state.data.tasks || []).forEach((task) => {
    const changed = syncTaskDeliveryRules(task);
    if (changed.length) task.updated_at = nowIso();
  });
}

function selectHtml(field, options, value) {
  const selected = String(value ?? "");
  return `<select data-field="${field}">${options.map(([id, label]) => `<option value="${escapeAttr(id)}" ${String(id) === selected ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select>`;
}

function groupTitle(id) {
  return state.data.groups.find((group) => group.id === id)?.title || "";
}

function specialTitle(id) {
  return id ? state.data.specials.find((special) => special.id === id)?.title || id : "普通事项";
}

function taskCountForSpecial(id) {
  return state.data.tasks.filter((task) => task.special_id === id).length;
}

function riskClass(value) {
  return ({ "高": "high", "中": "medium", "低": "low" })[value] || "medium";
}

function statusLabel(value) {
  return Object.fromEntries(STATUS_OPTIONS)[value] || value || "todo";
}

function statusClass(value) {
  return ({ done: "done", delayed: "delayed", blocked: "blocked", doing: "doing" })[value] || "todo";
}

function nowIso() {
  const date = new Date();
  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const hh = String(Math.floor(Math.abs(offset) / 60)).padStart(2, "0");
  const mm = String(Math.abs(offset) % 60).padStart(2, "0");
  return date.toISOString().replace(/\.\d{3}Z$/, `${sign}${hh}:${mm}`);
}

function toDay(value) {
  return new Date(value).toISOString().slice(0, 10);
}

function daysBetween(a, b) {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86400000);
}

function addDays(value, days) {
  const date = dateFromYmd(value);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function prevDay(value) {
  return addDays(value, -1);
}

function nextDay(value) {
  return addDays(value, 1);
}

function dateFromYmd(value) {
  const [year, month, day] = String(value).split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day));
}

function maxDate(a, b) {
  return a > b ? a : b;
}

function minDate(a, b) {
  return a < b ? a : b;
}

function formatMonthDay(value) {
  const [, month, day] = String(value || "").split("-");
  return month && day ? `${month}-${day}` : String(value || "");
}

function isYmd(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(value || "")) && !Number.isNaN(Date.parse(value));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

["q", "risk", "priority", "status"].forEach((id) => {
  $(`#${id}`).addEventListener(id === "q" ? "input" : "change", (event) => {
    state.filters[id] = event.target.value.trim();
    render();
  });
});
$("#refresh").addEventListener("click", load);
$("#clearFilters").addEventListener("click", clearFilters);
$("#editMode").addEventListener("click", enableEditMode);
$("#loginBtn").addEventListener("click", () => loginWorker().catch(showError));
$("#loginPassword").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  loginWorker().catch(showError);
});
$("#logout").addEventListener("click", logout);
$("#changePassword").addEventListener("click", () => changePassword().catch(showError));
$("#addTask").addEventListener("click", () => addTask().catch(showError));
$("#saveAll").addEventListener("click", () => saveAllTasks().catch(showError));
$("#addGroup").addEventListener("click", () => addGroup().catch(showError));
$("#addSpecial").addEventListener("click", () => addSpecial().catch(showError));
$("#addOperator").addEventListener("click", () => addOperator().catch(showError));
$("#addPerson").addEventListener("click", () => addPerson().catch(showError));
document.addEventListener("click", (event) => {
  if (!state.ownerFilterOpen || event.target.closest("[data-owner-filter]")) return;
  state.ownerFilterOpen = false;
  renderTableFilters();
});
document.addEventListener("click", (event) => {
  if (event.target.closest(".owner-editor,.operator-editor")) return;
  closeOwnerPickers(event);
});
document.addEventListener("click", (event) => {
  if (event.target.closest(".pr-quick-picker")) return;
  closePrSuggestMenus(event);
});
document.addEventListener("scroll", closeOwnerPickers, true);
document.addEventListener("scroll", closePrSuggestMenus, true);
document.addEventListener("scroll", positionOwnerFilterMenu, true);
window.addEventListener("resize", () => {
  closeOwnerPickers();
  closePrSuggestMenus();
  positionOwnerFilterMenu();
});

function showError(error) {
  $("#editStatus").textContent = `写入失败：${error.message}`;
  alert(error.message);
}

load().then(startRealtimeSync).catch((error) => {
  $("#meta").textContent = `读取失败：${error.message}`;
});
