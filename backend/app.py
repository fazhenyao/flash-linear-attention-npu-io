from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .db import (
        DB_PATH,
        ROOT,
        add_perf_model,
        apply_imported_perf_data,
        complete_perf_run,
        connect,
        export_all,
        find_perf_run,
        import_perf_cases,
        import_prof_dir,
        init_db,
        list_audit_entries,
        load_perf_data,
        load_repo_state_or_seed,
        make_id,
        recover_stale_perf_runs,
        seed_audit_if_empty,
        seed_if_empty,
        task_to_dict,
        trigger_perf_run,
        update_perf_run,
        update_task,
        upload_and_import_prof_dir,
        upsert_task,
    )
    from .perf_runner import (
        build_perf_cases_csv_download,
        build_perf_run_download,
        execute as execute_perf_test,
        list_prof_directories,
        match_prof_dir_from_paths,
        prof_tool_label,
        runner_status,
    )
    from .repo_store import persist_change
except ImportError:
    from db import (  # type: ignore
        DB_PATH,
        ROOT,
        add_perf_model,
        apply_imported_perf_data,
        complete_perf_run,
        connect,
        export_all,
        find_perf_run,
        import_perf_cases,
        import_prof_dir,
        init_db,
        list_audit_entries,
        load_perf_data,
        load_repo_state_or_seed,
        make_id,
        recover_stale_perf_runs,
        seed_audit_if_empty,
        seed_if_empty,
        task_to_dict,
        trigger_perf_run,
        update_perf_run,
        update_task,
        upload_and_import_prof_dir,
        upsert_task,
    )
    from perf_runner import (  # type: ignore
        build_perf_cases_csv_download,
        build_perf_run_download,
        execute as execute_perf_test,
        list_prof_directories,
        match_prof_dir_from_paths,
        prof_tool_label,
        runner_status,
    )
    from repo_store import persist_change  # type: ignore


FRONTEND = ROOT / "frontend"
_PERF_JOB_QUEUE: queue.Queue[tuple[str, dict]] = queue.Queue()
_PERF_WORKER_STARTED = False
_PERF_WORKER_LOCK = threading.Lock()
ALLOWED_TASK_FIELDS = {
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


class Handler(BaseHTTPRequestHandler):
    server_version = "FlashGanttIO/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed.path, parse_qs(parsed.query))
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        self.handle_write("POST")

    def do_PATCH(self) -> None:
        self.handle_write("PATCH")

    def do_DELETE(self) -> None:
        self.handle_write("DELETE")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        with connect() as conn:
            init_db(conn)
            seed_if_empty(conn)
            seed_audit_if_empty(conn)
            if path == "/api/health":
                self.send_json({"ok": True, "storage": "sqlite", "database": "data/project.sqlite3", "snapshot": "data/project-state.json", "auditLog": "data/audit-log.jsonl"})
            elif path == "/api/export":
                self.send_json(export_all(conn))
            elif path == "/api/audit":
                limit = int((query.get("limit") or ["10"])[0])
                q = (query.get("q") or [""])[0]
                self.send_json(list_audit_entries(conn, limit=limit, q=q))
            elif path == "/api/groups":
                rows = conn.execute("SELECT * FROM groups ORDER BY position, due_date").fetchall()
                self.send_json([dict(row) for row in rows])
            elif path == "/api/specials":
                rows = conn.execute("SELECT * FROM specials ORDER BY position, title").fetchall()
                self.send_json([dict(row) for row in rows])
            elif path == "/api/perf":
                self.send_json(load_perf_data(conn))
            elif path == "/api/perf/runner":
                self.send_json(runner_status())
            elif path == "/api/perf/cases/download":
                ids = [item for item in (query.get("ids") or [""])[0].split(",") if item.strip()]
                self.download_perf_cases(conn, ids)
            elif path == "/api/perf/prof-dirs":
                self.send_json({"dirs": list_prof_directories()})
            elif path.startswith("/api/perf/runs/") and path.endswith("/download"):
                run_id = path.removeprefix("/api/perf/runs/").removesuffix("/download")
                self.download_perf_run(conn, run_id)
            elif path == "/api/tasks":
                self.send_json(list_tasks(conn, query))
            elif path.startswith("/api/tasks/"):
                task_id = path.removeprefix("/api/tasks/")
                row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if not row:
                    self.send_error_json(404, "task not found")
                    return
                self.send_json(task_to_dict(conn, row))
            else:
                self.send_error_json(404, "api not found")

    def handle_write(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        with connect() as conn:
            init_db(conn)
            seed_if_empty(conn)
            seed_audit_if_empty(conn)
            try:
                if path == "/api/perf/cases/upload-prof" and method == "POST":
                    self.upload_prof_case(conn)
                    return
                payload = self.read_json()
                if path == "/api/tasks" and method == "POST":
                    self.create_task(conn, payload)
                elif path.startswith("/api/tasks/") and method == "PATCH":
                    self.patch_task(conn, path.removeprefix("/api/tasks/"), payload)
                elif path.startswith("/api/tasks/") and path.endswith("/split") and method == "POST":
                    task_id = path.split("/")[3]
                    self.split_task(conn, task_id, payload)
                elif path.startswith("/api/tasks/") and method == "DELETE":
                    task_id = path.removeprefix("/api/tasks/")
                    row = conn.execute("SELECT title FROM tasks WHERE id = ?", (task_id,)).fetchone()
                    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                    conn.commit()
                    self.finalize_change(conn, "task.delete", "task", task_id, "删除任务：" + (row["title"] if row else task_id))
                    self.send_json({"ok": True})
                elif path == "/api/perf/models" and method == "POST":
                    result = add_perf_model(conn, payload.get("model") or payload)
                    conn.commit()
                    self.send_json({"ok": True, "data": result}, status=201)
                elif path == "/api/perf/runs" and method == "POST":
                    result = trigger_perf_run(conn, payload, created_by="local")
                    conn.commit()
                    run = result.get("run") or {}
                    if run.get("status") == "queued":
                        enqueue_perf_run_job(run["id"], payload)
                    self.send_json({"ok": True, **result}, status=201)
                elif path == "/api/perf/cases/import" and method == "POST":
                    result = import_perf_cases(conn, payload)
                    conn.commit()
                    self.send_json({"ok": True, **result}, status=201)
                elif path == "/api/perf/cases/import-prof" and method == "POST":
                    result = import_prof_dir(conn, payload)
                    conn.commit()
                    self.send_json({"ok": True, **result}, status=201)
                elif path == "/api/perf/prof-dirs/match" and method == "POST":
                    prof_dir = match_prof_dir_from_paths(payload.get("paths") or [])
                    self.send_json({"ok": True, "prof_dir": prof_dir})
                elif path == "/api/groups" and method == "POST":
                    self.create_group(conn, payload)
                elif path.startswith("/api/groups/") and method == "PATCH":
                    self.patch_group(conn, path.removeprefix("/api/groups/"), payload)
                elif path.startswith("/api/groups/") and method == "DELETE":
                    self.delete_group(conn, path.removeprefix("/api/groups/"))
                elif path == "/api/specials" and method == "POST":
                    self.create_special(conn, payload)
                elif path.startswith("/api/specials/") and method == "PATCH":
                    self.patch_special(conn, path.removeprefix("/api/specials/"), payload)
                elif path.startswith("/api/specials/") and method == "DELETE":
                    self.delete_special(conn, path.removeprefix("/api/specials/"))
                else:
                    self.send_error_json(404, "api not found")
            except ValueError as exc:
                self.send_error_json(400, str(exc))
            except FileNotFoundError as exc:
                self.send_error_json(400, str(exc))
            except OSError as exc:
                self.send_error_json(400, str(exc))
            except RuntimeError as exc:
                self.send_error_json(500, "仓库同步失败：" + str(exc))

    def create_task(self, conn, payload: dict) -> None:
        task_id = payload.get("id") or make_id("task")
        payload = {key: payload.get(key) for key in ALLOWED_TASK_FIELDS if key in payload}
        payload.setdefault("title", "新任务")
        payload.setdefault("owner", "待排人力")
        payload.setdefault("status", "todo")
        payload.setdefault("risk", "中")
        payload.setdefault("priority", "P1")
        payload.setdefault("group_id", first_group_id(conn))
        payload.setdefault("start_date", now_iso()[:10])
        payload.setdefault("end_date", payload["start_date"])
        payload["id"] = task_id
        upsert_task(conn, payload)
        conn.commit()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        sync = self.finalize_change(conn, "task.create", "task", task_id, "新增任务：" + payload["title"], {"title": payload["title"]})
        body = task_to_dict(conn, row)
        body["_sync"] = sync
        self.send_json(body, status=201)

    def patch_task(self, conn, task_id: str, payload: dict) -> None:
        changes = {key: value for key, value in payload.items() if key in ALLOWED_TASK_FIELDS}
        if not update_task(conn, task_id, changes):
            raise ValueError("task not found or no valid fields")
        if "start_date" in changes or "end_date" in changes:
            row = conn.execute("SELECT start_date, end_date FROM tasks WHERE id = ?", (task_id,)).fetchone()
            conn.execute(
                """
                UPDATE task_segments
                SET start_date = ?, end_date = ?
                WHERE id = (
                  SELECT id FROM task_segments WHERE task_id = ? ORDER BY position LIMIT 1
                )
                """,
                (row["start_date"], row["end_date"], task_id),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        sync = self.finalize_change(conn, "task.update", "task", task_id, "更新任务：" + row["title"], {"fields": sorted(changes.keys())})
        body = task_to_dict(conn, row)
        body["_sync"] = sync
        self.send_json(body)

    def split_task(self, conn, task_id: str, payload: dict) -> None:
        break_start = payload.get("break_start")
        break_end = payload.get("break_end") or break_start
        reason = payload.get("reason") or "任务中断"
        if not break_start or not break_end:
            raise ValueError("break_start and break_end are required")
        segments = conn.execute(
            "SELECT * FROM task_segments WHERE task_id = ? ORDER BY position, start_date",
            (task_id,),
        ).fetchall()
        if not segments:
            raise ValueError("task segment not found")
        conn.execute("DELETE FROM task_segments WHERE task_id = ?", (task_id,))
        next_rows = []
        for seg in segments:
            start, end = seg["start_date"], seg["end_date"]
            if break_end < start or break_start > end:
                next_rows.append((start, end, seg["reason"]))
                continue
            if break_start > start:
                next_rows.append((start, prev_day(break_start), seg["reason"]))
            if break_end < end:
                next_rows.append((next_day(break_end), end, seg["reason"]))
        if not next_rows:
            raise ValueError("break range covers the whole task")
        for pos, (start, end, old_reason) in enumerate(next_rows):
            conn.execute(
                "INSERT INTO task_segments(id, task_id, start_date, end_date, reason, position) VALUES (?, ?, ?, ?, ?, ?)",
                (make_id("seg"), task_id, start, end, old_reason or reason, pos),
            )
        conn.execute("UPDATE tasks SET notes = ?, updated_at = datetime('now') WHERE id = ?", (reason, task_id))
        conn.commit()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        sync = self.finalize_change(conn, "task.split", "task", task_id, "切分任务：" + row["title"], {"break_start": break_start, "break_end": break_end, "reason": reason})
        body = task_to_dict(conn, row)
        body["_sync"] = sync
        self.send_json(body)

    def create_group(self, conn, payload: dict) -> None:
        group_id = payload.get("id") or make_id("group")
        title = payload.get("title") or "新分组"
        date = payload.get("due_date") or payload.get("end_date") or "2026-06-25"
        conn.execute(
            "INSERT INTO groups(id, title, due_date, start_date, end_date, position) VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, title, date, payload.get("start_date") or date, payload.get("end_date") or date, payload.get("position") or 999),
        )
        conn.commit()
        sync = self.finalize_change(conn, "group.create", "group", group_id, "新增分组：" + title)
        self.send_json({"ok": True, "id": group_id, "_sync": sync}, status=201)

    def patch_group(self, conn, group_id: str, payload: dict) -> None:
        allowed = {"title", "due_date", "start_date", "end_date", "position"}
        assignments = []
        values = []
        for key in allowed:
            if key in payload:
                assignments.append(f"{key} = ?")
                values.append(payload[key])
        if not assignments:
            raise ValueError("no valid fields")
        values.append(group_id)
        conn.execute(f"UPDATE groups SET {', '.join(assignments)} WHERE id = ?", values)
        conn.commit()
        row = conn.execute("SELECT title FROM groups WHERE id = ?", (group_id,)).fetchone()
        sync = self.finalize_change(conn, "group.update", "group", group_id, "更新分组：" + (row["title"] if row else group_id), {"fields": sorted(payload.keys())})
        self.send_json({"ok": True, "_sync": sync})

    def delete_group(self, conn, group_id: str) -> None:
        fallback = conn.execute("SELECT id FROM groups WHERE id != ? ORDER BY position LIMIT 1", (group_id,)).fetchone()
        if not fallback:
            raise ValueError("at least one group must remain")
        conn.execute("UPDATE tasks SET group_id = ? WHERE group_id = ?", (fallback["id"], group_id))
        conn.execute("UPDATE specials SET group_id = ? WHERE group_id = ?", (fallback["id"], group_id))
        conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
        conn.commit()
        self.finalize_change(conn, "group.delete", "group", group_id, "删除分组：" + group_id, {"fallback_group_id": fallback["id"]})
        self.send_json({"ok": True})

    def create_special(self, conn, payload: dict) -> None:
        special_id = payload.get("id") or make_id("special")
        conn.execute(
            "INSERT INTO specials(id, title, group_id, position, collapsed) VALUES (?, ?, ?, ?, 0)",
            (special_id, payload.get("title") or "专项：新专项", payload.get("group_id") or first_group_id(conn), payload.get("position") or 999),
        )
        conn.commit()
        title = payload.get("title") or "专项：新专项"
        sync = self.finalize_change(conn, "special.create", "special", special_id, "新增专项：" + title)
        self.send_json({"ok": True, "id": special_id, "_sync": sync}, status=201)

    def patch_special(self, conn, special_id: str, payload: dict) -> None:
        allowed = {"title", "group_id", "position", "collapsed"}
        assignments = []
        values = []
        for key in allowed:
            if key in payload:
                assignments.append(f"{key} = ?")
                values.append(payload[key])
        if not assignments:
            raise ValueError("no valid fields")
        values.append(special_id)
        conn.execute(f"UPDATE specials SET {', '.join(assignments)} WHERE id = ?", values)
        conn.commit()
        row = conn.execute("SELECT title FROM specials WHERE id = ?", (special_id,)).fetchone()
        sync = self.finalize_change(conn, "special.update", "special", special_id, "更新专项：" + (row["title"] if row else special_id), {"fields": sorted(payload.keys())})
        self.send_json({"ok": True, "_sync": sync})

    def delete_special(self, conn, special_id: str) -> None:
        conn.execute("UPDATE tasks SET special_id = NULL WHERE special_id = ?", (special_id,))
        conn.execute("DELETE FROM specials WHERE id = ?", (special_id,))
        conn.commit()
        self.finalize_change(conn, "special.delete", "special", special_id, "删除专项：" + special_id)
        self.send_json({"ok": True})

    def finalize_change(self, conn, action: str, entity: str, entity_id: str, summary: str, detail: dict | None = None) -> dict:
        return persist_change(conn, action, entity, entity_id, summary, detail)

    def serve_static(self, path: str) -> None:
        if path in {"/", "/io"}:
            path = "/index.html"
        target = (FRONTEND / path.lstrip("/")).resolve()
        if not str(target).startswith(str(FRONTEND.resolve())) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def upload_prof_case(self, conn) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("multipart/form-data required")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        prof_source = str(form.getfirst("prof_source", "") or "").strip()
        kernel_name = str(form.getfirst("kernel_name", "") or "").strip()
        archive_field = form["archive"] if "archive" in form else None
        if archive_field is not None and getattr(archive_field, "file", None) is not None and archive_field.filename:
            archive_bytes = archive_field.file.read()
            result = upload_and_import_prof_dir(
                conn,
                archive=archive_bytes,
                prof_source=prof_source,
                kernel_name=kernel_name,
            )
        else:
            file_fields = form["files"] if "files" in form else None
            if file_fields is None:
                raise ValueError("archive or files required")
            if not isinstance(file_fields, list):
                file_fields = [file_fields]
            entries: list[tuple[str, bytes]] = []
            for item in file_fields:
                if not getattr(item, "file", None) or not item.filename:
                    continue
                entries.append((item.filename, item.file.read()))
            result = upload_and_import_prof_dir(
                conn,
                files=entries,
                prof_source=prof_source,
                kernel_name=kernel_name,
            )
        conn.commit()
        self.send_json({"ok": True, **result}, status=201)

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"ok": False, "error": message}, status)

    def send_download(self, filename: str, data: bytes, content_type: str = "application/zip") -> None:
        from urllib.parse import quote

        safe_name = quote(filename)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"; filename*=UTF-8\'\'{safe_name}')
        self._send_cors_headers()
        self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
        self.end_headers()
        self.wfile.write(data)

    def download_perf_cases(self, conn, case_ids: list[str]) -> None:
        try:
            filename, payload = build_perf_cases_csv_download(load_perf_data(conn), case_ids)
            self.send_download(filename, payload)
        except ValueError as exc:
            self.send_error_json(400, str(exc))

    def download_perf_run(self, conn, run_id: str) -> None:
        try:
            data = load_perf_data(conn)
            run = find_perf_run(conn, run_id)
            if not run:
                self.send_error_json(404, "run not found")
                return
            filename, payload = build_perf_run_download(run, data)
            self.send_download(filename, payload)
        except ValueError as exc:
            self.send_error_json(400, str(exc))


def list_tasks(conn, query: dict[str, list[str]]) -> list[dict]:
    where = []
    values = []
    for field in ("risk", "priority", "owner", "status", "group_id", "special_id"):
        if query.get(field):
            where.append(f"{field} = ?")
            values.append(query[field][0])
    if query.get("q"):
        where.append("(title LIKE ? OR scope LIKE ? OR owner LIKE ?)")
        q = f"%{query['q'][0]}%"
        values.extend([q, q, q])
    sql = "SELECT * FROM tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY position, start_date, title"
    return [task_to_dict(conn, row) for row in conn.execute(sql, values).fetchall()]


def first_group_id(conn) -> str:
    row = conn.execute("SELECT id FROM groups ORDER BY position LIMIT 1").fetchone()
    if not row:
        raise ValueError("group not found")
    return row["id"]


def today_group_date(conn, group_id: str) -> str:
    row = conn.execute("SELECT due_date FROM groups WHERE id = ?", (group_id,)).fetchone()
    return row["due_date"] if row else "2026-06-25"


def prev_day(value: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def next_day(value: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def enqueue_perf_run_job(run_id: str, payload: dict) -> None:
    global _PERF_WORKER_STARTED
    with _PERF_WORKER_LOCK:
        if not _PERF_WORKER_STARTED:
            threading.Thread(target=_perf_run_worker, daemon=True, name="perf-run-worker").start()
            _PERF_WORKER_STARTED = True
    _PERF_JOB_QUEUE.put((run_id, payload))


def _perf_run_worker() -> None:
    while True:
        run_id, payload = _PERF_JOB_QUEUE.get()
        try:
            execute_perf_run_job(run_id, payload)
        finally:
            _PERF_JOB_QUEUE.task_done()


def execute_perf_run_job(run_id: str, payload: dict) -> None:
    try:
        from .db import now_iso
    except ImportError:
        from db import now_iso  # type: ignore

    conn = connect()
    try:
        init_db(conn)
        prof_tool = str(payload.get("prof_tool") or "msprof")
        update_perf_run(conn, run_id, status="running", message=f"正在执行 {prof_tool_label(prof_tool)}…")
        conn.commit()
        result = execute_perf_test(payload)
        if result.get("dry_run"):
            update_perf_run(
                conn,
                run_id,
                status="done",
                finished_at=now_iso(),
                message=result.get("message", "dry-run"),
                command=result.get("command", ""),
            )
            conn.commit()
            return
        apply_imported_perf_data(conn, result["data"])
        complete_perf_run(
            conn,
            run_id,
            snapshot=result["snapshot"],
            command=result.get("command", ""),
            message=result.get("message", "测试执行完成"),
            prof_dir=str(result.get("prof_dir") or ""),
            prof_source=str(result.get("prof_source") or result["snapshot"].get("prof_source") or ""),
        )
        conn.commit()
    except Exception as exc:
        try:
            update_perf_run(
                conn,
                run_id,
                status="failed",
                finished_at=now_iso(),
                message=str(exc),
            )
            conn.commit()
        except Exception as mark_exc:
            print(f"[perf] failed to mark run {run_id} as failed: {mark_exc} (original: {exc})")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    with connect() as conn:
        init_db(conn)
        load_repo_state_or_seed(conn)
        recovered = recover_stale_perf_runs(conn)
        conn.commit()
        if recovered:
            print(f"[perf] recovered {recovered} stale run(s)")

    if args.init_only:
        print(f"database ready: {DB_PATH}")
        return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Flash Gantt IO: http://{args.host}:{args.port}/io")
    server.serve_forever()


if __name__ == "__main__":
    main()
