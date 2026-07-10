from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "project.sqlite3"
STATE_PATH = ROOT / "data" / "project-state.json"
PERF_DATA_PATH = ROOT / "data" / "performance-data.json"
DOCS_PERF_DATA_PATH = ROOT / "docs" / "performance-data.json"
AUDIT_LOG_PATH = ROOT / "data" / "audit-log.jsonl"
PROJECT_DATA_CANDIDATES = [
    ROOT / "data" / "seed-project-data.json",
    ROOT.parent / "flash-linear-attention-npu-project" / "project-data.json",
]
GANTT_HTML_CANDIDATES = [
    ROOT / "data" / "seed-gantt-view.html",
    ROOT.parent / "flash-linear-attention-npu-project" / "gantt-view.html",
]
BASE_DATE = datetime(2026, 6, 15, tzinfo=timezone(timedelta(hours=8))).date()

RISK_TO_PRIORITY = {"高": "P0", "中": "P1", "低": "P2"}
DEFAULT_PROJECT = {
    "name": "flash-linear-attention-npu",
    "repository": "https://github.com/flashserve/flash-linear-attention-npu",
    "baselineDate": "2026-06-15",
    "projectOwner": {"name": "待填写", "email": "待填写"},
}


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def as_json(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def parse_json(value: str | None, fallback: Any = None) -> Any:
    if value in (None, ""):
        return [] if fallback is None else fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback if fallback is not None else value


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS project_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS groups (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          due_date TEXT NOT NULL,
          start_date TEXT NOT NULL,
          end_date TEXT NOT NULL,
          position INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS specials (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          group_id TEXT,
          position INTEGER NOT NULL DEFAULT 0,
          collapsed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          scope TEXT NOT NULL DEFAULT '',
          target TEXT NOT NULL DEFAULT '',
          owner TEXT NOT NULL DEFAULT '待排人力',
          status TEXT NOT NULL DEFAULT 'todo',
          risk TEXT NOT NULL DEFAULT '中',
          priority TEXT NOT NULL DEFAULT 'P1',
          group_id TEXT NOT NULL,
          special_id TEXT,
          start_date TEXT NOT NULL,
          end_date TEXT NOT NULL,
          evidence TEXT NOT NULL DEFAULT '[]',
          dependencies TEXT NOT NULL DEFAULT '[]',
          notes TEXT NOT NULL DEFAULT '',
          position INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(group_id) REFERENCES groups(id) ON UPDATE CASCADE,
          FOREIGN KEY(special_id) REFERENCES specials(id) ON UPDATE CASCADE ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS task_segments (
          id TEXT PRIMARY KEY,
          task_id TEXT NOT NULL,
          start_date TEXT NOT NULL,
          end_date TEXT NOT NULL,
          reason TEXT NOT NULL DEFAULT '',
          position INTEGER NOT NULL DEFAULT 0,
          FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          action TEXT NOT NULL,
          entity TEXT NOT NULL,
          entity_id TEXT NOT NULL,
          summary TEXT NOT NULL,
          detail TEXT NOT NULL DEFAULT '{}',
          source TEXT NOT NULL DEFAULT 'backend'
        );
        """
    )
    conn.commit()


def seed_if_empty(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    if count:
        return
    if STATE_PATH.exists():
        import_state_snapshot(conn, STATE_PATH)
        conn.commit()
        return
    import_project_data(conn, first_existing(PROJECT_DATA_CANDIDATES))
    gantt_html = first_existing(GANTT_HTML_CANDIDATES)
    if gantt_html:
        import_gantt_html_overlay(conn, gantt_html)
    conn.commit()


def load_repo_state_or_seed(conn: sqlite3.Connection) -> None:
    # Backward-compatible entrypoint: SQLite is the source of truth now.
    # Repository JSON is only used for first boot or rebuild.
    seed_if_empty(conn)
    seed_audit_if_empty(conn)


def seed_audit_if_empty(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM audit_entries").fetchone()[0]
    if count or not AUDIT_LOG_PATH.exists():
        return
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            insert_audit_entry(conn, entry)
    conn.commit()


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def import_state_snapshot(conn: sqlite3.Connection, path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    conn.execute("DELETE FROM task_segments")
    conn.execute("DELETE FROM tasks")
    conn.execute("DELETE FROM specials")
    conn.execute("DELETE FROM groups")
    conn.execute("DELETE FROM project_meta")

    conn.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
        ("project", as_json(data.get("project", DEFAULT_PROJECT))),
    )
    conn.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
        ("repoScan", as_json(data.get("repoScan", {}))),
    )

    for pos, group in enumerate(data.get("groups", [])):
        conn.execute(
            """
            INSERT OR REPLACE INTO groups(id, title, due_date, start_date, end_date, position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                group.get("id") or make_id("group"),
                group.get("title") or "未命名分组",
                group.get("due_date") or group.get("end_date") or "2026-06-25",
                group.get("start_date") or group.get("due_date") or "2026-06-25",
                group.get("end_date") or group.get("due_date") or "2026-06-25",
                int(group.get("position", pos)),
            ),
        )

    for pos, special in enumerate(data.get("specials", [])):
        conn.execute(
            """
            INSERT OR REPLACE INTO specials(id, title, group_id, position, collapsed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                special.get("id") or make_id("special"),
                special.get("title") or "专项：未命名",
                special.get("group_id"),
                int(special.get("position", pos)),
                int(bool(special.get("collapsed", 0))),
            ),
        )

    for pos, task in enumerate(data.get("tasks", [])):
        payload = {
            "id": task.get("id") or make_id("task"),
            "title": task.get("title") or "未命名任务",
            "scope": task.get("scope", ""),
            "target": task.get("target", ""),
            "owner": task.get("owner") or "待排人力",
            "status": task.get("status") or "todo",
            "risk": task.get("risk") or "中",
            "priority": task.get("priority") or RISK_TO_PRIORITY.get(task.get("risk"), "P1"),
            "group_id": task.get("group_id"),
            "special_id": task.get("special_id") or None,
            "start_date": task.get("start_date"),
            "end_date": task.get("end_date"),
            "evidence": task.get("evidence", []),
            "dependencies": task.get("dependencies", []),
            "notes": task.get("notes", ""),
            "position": int(task.get("position", pos)),
        }
        upsert_task(conn, payload)
        if task.get("created_at") or task.get("updated_at"):
            conn.execute(
                "UPDATE tasks SET created_at = COALESCE(?, created_at), updated_at = COALESCE(?, updated_at) WHERE id = ?",
                (task.get("created_at"), task.get("updated_at"), payload["id"]),
            )
        conn.execute("DELETE FROM task_segments WHERE task_id = ?", (payload["id"],))
        for seg_pos, segment in enumerate(task.get("segments") or []):
            conn.execute(
                """
                INSERT INTO task_segments(id, task_id, start_date, end_date, reason, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.get("id") or make_id("seg"),
                    payload["id"],
                    segment.get("start_date") or payload["start_date"],
                    segment.get("end_date") or payload["end_date"],
                    segment.get("reason", ""),
                    int(segment.get("position", seg_pos)),
                ),
            )


def import_project_data(conn: sqlite3.Connection, path: Path | None) -> None:
    data = DEFAULT_PROJECT
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8-sig"))

    project = data.get("project", DEFAULT_PROJECT)
    conn.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
        ("project", as_json(project)),
    )

    repo_risk = {
        item.get("id"): item
        for item in data.get("repoScan", {}).get("items", [])
        if item.get("id")
    }
    repo_status = data.get("repoScan", {})
    conn.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
        ("repoScan", as_json(repo_status)),
    )

    milestones = data.get("milestones") or [
        {"id": "M-2026-06-18", "title": "06-18 转测", "dueDate": "2026-06-18", "items": []},
        {"id": "M-2026-06-25", "title": "06-25 转测", "dueDate": "2026-06-25", "items": []},
        {"id": "M-2026-07-15", "title": "07-15 转测", "dueDate": "2026-07-15", "items": []},
    ]

    for group_pos, milestone in enumerate(milestones):
        group_id = milestone.get("id") or make_id("group")
        title = milestone.get("title") or group_id
        due_date = milestone.get("dueDate") or "2026-06-25"
        conn.execute(
            """
            INSERT OR REPLACE INTO groups(id, title, due_date, start_date, end_date, position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (group_id, title, due_date, due_date, due_date, group_pos),
        )

    default_specials = [
        ("docs", "专项：对应场景文档更新", "M-2026-06-25", 90),
        ("launch", "专项：<<<>>> 调用示例", "M-2026-06-25", 91),
    ]
    for row in default_specials:
        conn.execute(
            "INSERT OR REPLACE INTO specials(id, title, group_id, position) VALUES (?, ?, ?, ?)",
            row,
        )

    for milestone in milestones:
        group_id = milestone.get("id")
        due_date = milestone.get("dueDate") or "2026-06-25"
        for pos, item in enumerate(milestone.get("items", [])):
            task_id = item.get("id") or make_id("task")
            scan = repo_risk.get(task_id, {})
            risk = scan.get("repoRisk") or item.get("risk") or "中"
            priority = item.get("priority") or RISK_TO_PRIORITY.get(risk, "P1")
            title = item.get("title") or task_id
            special_id = infer_special_id(title)
            evidence = item.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if scan.get("evidence"):
                evidence.append(scan["evidence"])
            upsert_task(
                conn,
                {
                    "id": task_id,
                    "title": title,
                    "scope": item.get("scope", ""),
                    "target": item.get("target", ""),
                    "owner": item.get("owner") or "待排人力",
                    "status": item.get("status") or "todo",
                    "risk": risk,
                    "priority": priority,
                    "group_id": group_id,
                    "special_id": special_id,
                    "start_date": due_date,
                    "end_date": due_date,
                    "evidence": evidence,
                    "dependencies": item.get("dependencies", []),
                    "notes": "",
                    "position": pos,
                },
            )


def infer_special_id(title: str) -> str | None:
    if "对应场景" in title or "文档更新" in title:
        return "docs"
    if "<<<>>>" in title or "调用示例" in title or "方式调用" in title:
        return "launch"
    return None


def import_gantt_html_overlay(conn: sqlite3.Connection, path: Path) -> None:
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="initialTasks">(.+?)</script>', html, re.S)
    if not match:
        return
    tasks = json.loads(match.group(1))
    conn.execute("DELETE FROM task_segments")
    conn.execute("DELETE FROM tasks")
    group_by_title = {
        row["title"]: row["id"]
        for row in conn.execute("SELECT id, title FROM groups").fetchall()
    }
    for pos, task in enumerate(tasks):
        group_title = task.get("group") or "06-25 转测"
        group_id = group_by_title.get(group_title)
        if not group_id:
            group_id = make_id("group")
            group_by_title[group_title] = group_id
            due = date_from_index(task.get("start", 0) + task.get("span", 1) - 1)
            conn.execute(
                """
                INSERT INTO groups(id, title, due_date, start_date, end_date, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (group_id, group_title, due, due, due, len(group_by_title)),
            )
        start = date_from_index(int(task.get("start", 0)))
        end = date_from_index(int(task.get("start", 0)) + int(task.get("span", 1)) - 1)
        payload = {
            "id": task["id"],
            "title": task.get("title", ""),
            "owner": task.get("owner", "待排人力"),
            "risk": task.get("risk", "中"),
            "priority": task.get("priority", RISK_TO_PRIORITY.get(task.get("risk", "中"), "P1")),
            "group_id": group_id,
            "special_id": task.get("special"),
            "start_date": start,
            "end_date": end,
            "position": pos,
        }
        existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
        if existing:
            update_task(conn, task["id"], payload)
        else:
            payload.update({"scope": "", "target": "", "status": "todo", "evidence": [], "dependencies": [], "notes": ""})
            upsert_task(conn, payload)

        conn.execute("DELETE FROM task_segments WHERE task_id = ?", (task["id"],))
        for seg_pos, segment in enumerate(task.get("segments") or [{"start": task.get("start", 0), "span": task.get("span", 1)}]):
            seg_start = int(segment.get("start", task.get("start", 0)))
            seg_end = seg_start + int(segment.get("span", 1)) - 1
            conn.execute(
                """
                INSERT INTO task_segments(id, task_id, start_date, end_date, position)
                VALUES (?, ?, ?, ?, ?)
                """,
                (make_id("seg"), task["id"], date_from_index(seg_start), date_from_index(seg_end), seg_pos),
            )


def date_from_index(index: int) -> str:
    return (BASE_DATE + timedelta(days=index)).isoformat()


def upsert_task(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    timestamp = now_iso()
    conn.execute(
        """
        INSERT OR REPLACE INTO tasks(
          id, title, scope, target, owner, status, risk, priority, group_id, special_id,
          start_date, end_date, evidence, dependencies, notes, position, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM tasks WHERE id = ?), ?), ?)
        """,
        (
            payload["id"],
            payload.get("title", ""),
            payload.get("scope", ""),
            payload.get("target", ""),
            payload.get("owner", "待排人力"),
            payload.get("status", "todo"),
            payload.get("risk", "中"),
            payload.get("priority", "P1"),
            payload.get("group_id"),
            payload.get("special_id") or None,
            payload.get("start_date"),
            payload.get("end_date"),
            as_json(payload.get("evidence", [])),
            as_json(payload.get("dependencies", [])),
            payload.get("notes", ""),
            int(payload.get("position", 0)),
            payload["id"],
            timestamp,
            timestamp,
        ),
    )
    if not conn.execute("SELECT 1 FROM task_segments WHERE task_id = ?", (payload["id"],)).fetchone():
        conn.execute(
            "INSERT INTO task_segments(id, task_id, start_date, end_date, position) VALUES (?, ?, ?, ?, 0)",
            (make_id("seg"), payload["id"], payload.get("start_date"), payload.get("end_date")),
        )


def update_task(conn: sqlite3.Connection, task_id: str, changes: dict[str, Any]) -> bool:
    allowed = {
        "title",
        "scope",
        "target",
        "owner",
        "status",
        "risk",
        "priority",
        "group_id",
        "special_id",
        "start_date",
        "end_date",
        "evidence",
        "dependencies",
        "notes",
        "position",
    }
    fields = []
    values = []
    for key, value in changes.items():
        if key not in allowed:
            continue
        if key in {"evidence", "dependencies"}:
            value = as_json(value)
        fields.append(f"{key} = ?")
        values.append(value)
    if not fields:
        return False
    fields.append("updated_at = ?")
    values.append(now_iso())
    values.append(task_id)
    cur = conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values)
    return cur.rowcount > 0


def task_to_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    task = dict(row)
    task["evidence"] = parse_json(task.get("evidence"), [])
    task["dependencies"] = parse_json(task.get("dependencies"), [])
    task["segments"] = [
        dict(seg)
        for seg in conn.execute(
            "SELECT id, start_date, end_date, reason, position FROM task_segments WHERE task_id = ? ORDER BY position, start_date",
            (task["id"],),
        ).fetchall()
    ]
    return task


def export_all(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "storageVersion": 1,
        "generatedAt": now_iso(),
        "project": parse_json(conn.execute("SELECT value FROM project_meta WHERE key = 'project'").fetchone()["value"], {}),
        "repoScan": parse_json(conn.execute("SELECT value FROM project_meta WHERE key = 'repoScan'").fetchone()["value"], {}),
        "groups": [dict(row) for row in conn.execute("SELECT * FROM groups ORDER BY position, due_date").fetchall()],
        "specials": [dict(row) for row in conn.execute("SELECT * FROM specials ORDER BY position, title").fetchall()],
        "tasks": [
            task_to_dict(conn, row)
            for row in conn.execute("SELECT * FROM tasks ORDER BY position, start_date, title").fetchall()
        ],
    }


def write_state_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = export_all(conn)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def insert_audit_entry(conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO audit_entries(ts, action, entity, entity_id, summary, detail, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.get("ts") or now_iso(),
            entry.get("action") or "",
            entry.get("entity") or "",
            entry.get("id") or entry.get("entity_id") or "",
            entry.get("summary") or "",
            as_json(entry.get("detail", {})),
            entry.get("source") or "backend",
        ),
    )


def audit_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "ts": row["ts"],
        "action": row["action"],
        "entity": row["entity"],
        "id": row["entity_id"],
        "summary": row["summary"],
        "detail": parse_json(row["detail"], {}),
        "source": row["source"],
    }


def list_audit_entries(conn: sqlite3.Connection, limit: int = 10, q: str = "") -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    values: list[Any] = []
    where = ""
    if q:
        where = """
        WHERE summary LIKE ?
           OR action LIKE ?
           OR entity_id LIKE ?
           OR detail LIKE ?
        """
        like = f"%{q}%"
        values.extend([like, like, like, like])
    values.append(limit)
    rows = conn.execute(
        f"""
        SELECT ts, action, entity, entity_id, summary, detail, source
        FROM audit_entries
        {where}
        ORDER BY id DESC
        LIMIT ?
        """,
        values,
    ).fetchall()
    return [audit_row_to_dict(row) for row in rows]


def write_audit_snapshot(conn: sqlite3.Connection) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """
        SELECT ts, action, entity, entity_id, summary, detail, source
        FROM audit_entries
        ORDER BY id
        """
    ).fetchall()
    with AUDIT_LOG_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(audit_row_to_dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_perf_data(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT value FROM project_meta WHERE key = 'perfData'").fetchone()
    if row:
        return parse_json(row["value"], default_perf_data())
    if PERF_DATA_PATH.exists():
        data = json.loads(PERF_DATA_PATH.read_text(encoding="utf-8-sig"))
        save_perf_data(conn, data)
        return data
    data = default_perf_data()
    save_perf_data(conn, data)
    return data


def save_perf_data(conn: sqlite3.Connection, data: dict[str, Any]) -> dict[str, Any]:
    data = normalize_perf_data(data)
    conn.execute(
        "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
        ("perfData", as_json(data)),
    )
    PERF_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERF_DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOCS_PERF_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PERF_DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def default_perf_data() -> dict[str, Any]:
    if PERF_DATA_PATH.exists():
        return json.loads(PERF_DATA_PATH.read_text(encoding="utf-8-sig"))
    return {
        "version": now_iso(),
        "models": [{"id": "gdn", "label": "GDN", "position": 0, "active": True}],
        "cases": [],
        "snapshots": [],
        "runs": [],
    }


def normalize_perf_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": data.get("version") or now_iso(),
        "models": list(data.get("models") or []),
        "cases": list(data.get("cases") or []),
        "snapshots": list(data.get("snapshots") or []),
        "runs": list(data.get("runs") or []),
    }


def add_perf_model(conn: sqlite3.Connection, model: dict[str, Any]) -> dict[str, Any]:
    data = load_perf_data(conn)
    model_id = model.get("id") or make_id("model")
    if any(item.get("id") == model_id for item in data["models"]):
        raise ValueError("model already exists")
    data["models"].append({
        "id": model_id,
        "label": model.get("label") or model_id,
        "position": len(data["models"]),
        "active": True,
    })
    data["version"] = now_iso()
    return save_perf_data(conn, data)


def trigger_perf_run(conn: sqlite3.Connection, payload: dict[str, Any], created_by: str = "backend") -> dict[str, Any]:
    try:
        from .perf_runner import TRIGGER_SCRIPTS, build_command, ensure_runner_configured, load_config, resolve_npu_device
    except ImportError:
        from perf_runner import TRIGGER_SCRIPTS, build_command, ensure_runner_configured, load_config, resolve_npu_device  # type: ignore

    ensure_runner_configured()
    return create_queued_perf_run(conn, payload, created_by, build_command(payload))


def resolve_run_device(payload: dict[str, Any]) -> str:
    try:
        try:
            from .perf_runner import load_config, resolve_npu_device
        except ImportError:
            from perf_runner import load_config, resolve_npu_device  # type: ignore
        return str(resolve_npu_device(payload, load_config()))
    except (ImportError, ValueError, TypeError):
        raw = payload.get("device")
        if raw is not None and str(raw).strip() != "":
            return str(int(raw))
        return "2"


def resolve_run_chip(payload: dict[str, Any]) -> str:
    try:
        try:
            from .perf_runner import load_config, resolve_chip
        except ImportError:
            from perf_runner import load_config, resolve_chip  # type: ignore
        return resolve_chip(payload, load_config())
    except (ImportError, ValueError, TypeError):
        raw = str(payload.get("chip") or "").strip().upper()
        return raw if raw in {"A2", "A3"} else "A2"


def create_queued_perf_run(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    created_by: str,
    command: str,
) -> dict[str, Any]:
    data = load_perf_data(conn)
    case_id = str(payload.get("case_id") or "").strip()
    model_id = payload.get("model_id")
    chip = resolve_run_chip(payload)
    if chip not in {"A2", "A3"}:
        raise ValueError("chip must be A2 or A3")
    if case_id and not any(item.get("id") == case_id for item in data["cases"]):
        raise ValueError("case not found")
    if not any(item.get("id") == model_id for item in data["models"]):
        raise ValueError("model not found")
    timestamp = now_iso()
    prof_tool = str(payload.get("prof_tool") or "msprof").strip()
    run = {
        "id": make_id("run"),
        "case_id": case_id,
        "model_id": model_id,
        "chip": chip,
        "device": resolve_run_device(payload),
        "attributes": payload.get("attributes") or {},
        "prof_tool": prof_tool,
        "script_path": str(payload.get("script_path") or TRIGGER_SCRIPTS[0]["id"]),
        "kernel_name": payload.get("kernel_name") or "",
        "status": "queued",
        "snapshot_id": "",
        "created_by": created_by,
        "created_at": timestamp,
        "finished_at": "",
        "message": f"已排队，等待执行 {prof_tool}",
        "command": command,
    }
    data["runs"].insert(0, run)
    data["version"] = timestamp
    save_perf_data(conn, data)
    return {"data": data, "run": run, "execution_mode": "real"}


def update_perf_run(conn: sqlite3.Connection, run_id: str, **fields: Any) -> dict[str, Any]:
    data = load_perf_data(conn)
    updated = None
    for index, run in enumerate(data.get("runs", [])):
        if run.get("id") == run_id:
            data["runs"][index] = {**run, **fields}
            updated = data["runs"][index]
            break
    if updated is None:
        raise ValueError(f"run not found: {run_id}")
    data["version"] = now_iso()
    save_perf_data(conn, data)
    return updated


def complete_perf_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    snapshot: dict[str, Any],
    command: str,
    message: str,
) -> dict[str, Any]:
    data = load_perf_data(conn)
    for index, run in enumerate(data.get("runs", [])):
        if run.get("id") != run_id:
            continue
        data["runs"][index] = {
            **run,
            "case_id": snapshot.get("case_id", run.get("case_id")),
            "status": "done",
            "snapshot_id": snapshot["id"],
            "finished_at": now_iso(),
            "message": message,
            "command": command,
            "prof_tool": snapshot.get("prof_tool", run.get("prof_tool", "msprof")),
            "kernel_name": snapshot.get("kernel_name", run.get("kernel_name", "")),
            "snapshot": snapshot,
        }
        break
    else:
        raise ValueError(f"run not found: {run_id}")
    data["version"] = now_iso()
    save_perf_data(conn, data)
    return {"data": data, "run": next(item for item in data["runs"] if item.get("id") == run_id)}


def apply_imported_perf_data(conn: sqlite3.Connection, imported_data: dict[str, Any]) -> dict[str, Any]:
    data = load_perf_data(conn)
    data["cases"] = imported_data.get("cases", data.get("cases", []))
    data["snapshots"] = imported_data.get("snapshots", data.get("snapshots", []))
    imported_run_ids = {item.get("id") for item in imported_data.get("runs", [])}
    preserved_runs = [item for item in data.get("runs", []) if item.get("id") not in imported_run_ids]
    data["runs"] = preserved_runs
    data["version"] = now_iso()
    return save_perf_data(conn, data)
