#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


class DashboardHandler(SimpleHTTPRequestHandler):
    backend_host = "127.0.0.1"
    backend_port = 8787

    def do_GET(self) -> None:
        if urlsplit(self.path).path.startswith("/api/"):
            self.proxy_api()
            return
        super().do_GET()

    def do_POST(self) -> None:
        self.proxy_api()

    def do_PATCH(self) -> None:
        self.proxy_api()

    def do_DELETE(self) -> None:
        self.proxy_api()

    def do_OPTIONS(self) -> None:
        self.proxy_api()

    def proxy_api(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "connection", "content-length"}
        }
        connection = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=300)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "content-length", "transfer-encoding"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError as exc:
            message = f'{{"ok":false,"error":"local backend unavailable: {exc}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
        finally:
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the performance dashboard with a same-origin API proxy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8787)
    args = parser.parse_args()

    DashboardHandler.backend_host = args.backend_host
    DashboardHandler.backend_port = args.backend_port
    handler = lambda *handler_args, **kwargs: DashboardHandler(  # noqa: E731
        *handler_args, directory=str(ROOT / "docs"), **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"dashboard: http://{args.host}:{args.port}/performance-dashboard.html")
    print(f"api proxy: /api/* -> http://{args.backend_host}:{args.backend_port}/api/*")
    server.serve_forever()


if __name__ == "__main__":
    main()
