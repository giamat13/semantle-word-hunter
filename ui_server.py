"""Local web UI for the solver. Standard library only.

Serves ui.html, streams the solver's events to the browser over Server-Sent Events, and lets the page
start and stop a run. Bound to 127.0.0.1; POST requests must come from the page itself.
"""

import argparse
import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_PATH = Path(__file__).with_name("ui.html")


class Hub:
    """Holds the event log of the current run and runs one game at a time in a worker thread."""

    def __init__(self, base_args: argparse.Namespace, hooks: dict):
        self.base_args = base_args
        self.hooks = hooks
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.lock = threading.Lock()
        self.running = False
        self.stop = threading.Event()
        self.pending: dict | None = None  # a run requested while another was still stopping
        self.last_args = base_args  # arguments of the latest run: feedback goes to the file that run used
        self.live: dict = {}  # model / supervisor of the running game; play() reads it every round

    def emit(self, ev: dict) -> None:
        with self.cond:
            self.events.append({**ev, "t": time.time()})
            self.cond.notify_all()

    def replay_start(self) -> int:
        """Index of the last 'reset' event, so a page that connects late replays only the current run."""
        with self.cond:
            for i in range(len(self.events) - 1, -1, -1):
                if self.events[i]["type"] == "reset":
                    return i
        return 0

    def start_run(self, overrides: dict) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.stop = threading.Event()
        run_args = argparse.Namespace(**vars(self.base_args))
        for key, value in overrides.items():
            setattr(run_args, key, value)
        self.last_args = run_args
        self.emit({"type": "reset"})
        self.live = {"model": run_args.model, "supervisor": run_args.supervisor}
        threading.Thread(target=self._run, args=(run_args, self.stop, self.live), daemon=True).start()
        return True

    def set_live(self, changes: dict) -> bool:
        """Switch model and/or supervisor of the running game; applies from the next round."""
        with self.lock:
            if not self.running:
                return False
            self.live.update(changes)
            return True

    def request_run(self, overrides: dict) -> str:
        """Run now. If a game is in progress, stop it and start the new one as soon as it has stopped."""
        with self.lock:
            busy = self.running
            if busy:
                self.pending = overrides
                self.stop.set()
        if busy:
            self.emit({"type": "restarting"})
            return "restarting"
        self.start_run(overrides)
        return "started"

    def _run(self, args, stop: threading.Event, live: dict) -> None:
        console = self.hooks["emit"]

        def emit(ev: dict) -> None:
            self.emit(ev)
            console(ev)

        code = 2
        try:
            code = self.hooks["play"](args, emit, stop, live)
        except Exception as e:  # keep the UI informed instead of dying silently in the thread
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            with self.lock:
                self.running = False
                pending, self.pending = self.pending, None
            self.emit({"type": "finished", "code": code})
            if pending is not None:
                self.start_run(pending)

    def state(self) -> dict:
        b = self.base_args
        return {
            "running": self.running,
            "defaults": {"model": b.model, "supervisor": b.supervisor, "batch": b.batch, "backend": b.backend,
                         "subagents": b.subagents, "subagent_model": b.subagent_model, "test": b.test},
            "live": dict(self.live),
            "knowledge": self.hooks["load_knowledge"](),
        }


MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\]:/@]{0,119}$")  # alias, model ID or provider:model, including future ones


def clean_models(raw: dict) -> dict:
    out = {}
    for key in ("model", "supervisor", "subagent_model"):
        value = raw.get(key)
        if isinstance(value, str) and MODEL_NAME.match(value) and not (key == "model" and value == "off"):
            out[key] = value
    return out


def clean_overrides(raw: dict, hooks: dict) -> dict:
    """Validates what the page sends; anything unexpected is ignored."""
    out: dict = clean_models(raw)
    if raw.get("backend") in ("auto", "cli", "api"):
        out["backend"] = raw["backend"]
    if isinstance(raw.get("batch"), int):
        out["batch"] = max(1, min(10, raw["batch"]))
    if isinstance(raw.get("max_guesses"), int):
        out["max_guesses"] = max(0, raw["max_guesses"])
    if isinstance(raw.get("test"), bool):
        out["test"] = raw["test"]
    if raw.get("subagents") in (0, 1, 2, 3):
        out["subagents"] = raw["subagents"]
    out["replay"] = True  # pressing run always runs, even when today's puzzle is already solved
    return out


def make_handler(hub: Hub, hooks: dict, port: int):
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console for solver output
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            path = self.path.split("?", 1)[0]  # the page accepts ?theme=dark|light
            if path in ("/", "/index.html"):
                return self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, hub.state())
            if path == "/api/models":
                return self._json(200, hooks["model_options"]())
            if path == "/api/providers":
                return self._json(200, hooks["provider_status"]())
            if path == "/events":
                return self._stream()
            self._json(404, {"error": "not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self._host_ok() or (origin and origin.removeprefix("http://") not in allowed):
                return self._json(403, {"error": "forbidden"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                raw = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if self.path == "/api/start":
                status = hub.request_run(clean_overrides(raw if isinstance(raw, dict) else {}, hooks))
                return self._json(200, {"status": status})
            if self.path == "/api/feedback":
                body = raw if isinstance(raw, dict) else {}
                rating = body.get("rating") if body.get("rating") in ("low", "ok", "high") else ""
                text = body.get("text") if isinstance(body.get("text"), str) else ""
                return self._json(200, {"saved": hooks["feedback"](hub.last_args, rating, text[:600])})
            if self.path == "/api/live":
                changes = clean_models(raw if isinstance(raw, dict) else {})
                return self._json(200, {"applied": hub.set_live(changes), "changes": changes})
            if self.path == "/api/stop":
                hub.stop.set()
                return self._json(200, {"stopping": True})
            self._json(404, {"error": "not found"})

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            idx = hub.replay_start()
            try:
                while True:
                    with hub.cond:
                        if idx >= len(hub.events):
                            hub.cond.wait(15)
                        batch = hub.events[idx:]
                        idx += len(batch)
                    if not batch:
                        self.wfile.write(b": keepalive\n\n")
                    for ev in batch:
                        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

    return Handler


def serve(args: argparse.Namespace, hooks: dict) -> int:
    hub = Hub(args, hooks)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub, hooks, port))
            break
        except OSError:
            continue
    if server is None:
        print(f"No free port in {args.port}-{args.port + 19}")
        return 1
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"Semantle solver UI: {url}   (Ctrl-C or stop debugging to quit)")
    if args.autostart:
        hub.start_run({"replay": True})
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        hub.stop.set()
    finally:
        server.server_close()
    return 0
