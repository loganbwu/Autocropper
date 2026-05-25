#!/usr/bin/env python3

import argparse
import base64
import queue
import subprocess
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .main import (
    compute_crop,
    get_capture_time,
    has_existing_crop,
    load_models,
    write_xmp,
)

PREFETCH = 10  # images to pre-process ahead; bounds peak RAM to ~PREFETCH × preview size

# Models load eagerly in background so they're ready when the user picks a folder
_models = None
_models_ready = threading.Event()


def _load_models_thread():
    global _models
    print("Loading AI model in background...")
    _models = load_models()
    print("AI model ready.")
    _models_ready.set()


class ReviewState:
    """Producer/consumer design: producer pre-processes up to PREFETCH images ahead,
    consumer presents them and waits for decisions. The user rarely waits."""

    def __init__(self, files, models, force=False, all_people=False):
        self.files = files
        self.models = models
        self.force = force
        self.all_people = all_people

        self.accepted = 0
        self.rejected = 0
        self.skipped = 0
        self.status = "loading"
        self.current = None

        # Total files that will need review (excludes already-cropped upfront).
        # Decremented further as no-person or no-difference detections are found.
        self.total_eligible = sum(
            1 for f in files if force or not has_existing_crop(f)
        )

        # Only processed results enter the queue (skips handled inline by producer).
        # maxsize bounds memory: each slot ≈ one preview JPEG pair (~1–4 MB).
        self._prefetch_q = queue.Queue(maxsize=PREFETCH)
        self._decision_q = queue.Queue()
        self._lock = threading.Lock()

        threading.Thread(target=self._producer, daemon=True).start()
        threading.Thread(target=self._consumer, daemon=True).start()

    def _producer(self):
        try:
            for cr3 in self.files:
                if not self.force and has_existing_crop(cr3):
                    with self._lock:
                        self.skipped += 1
                    continue

                try:
                    result = compute_crop(self.models, cr3, self.all_people)
                except Exception as e:
                    print(f"  Warning: skipping {cr3.name} — {e}")
                    with self._lock:
                        self.skipped += 1
                    continue

                if result is None:
                    print(f"  No person: {cr3.name}")
                    with self._lock:
                        self.skipped += 1
                    continue

                print(f"  Ready:     {cr3.name}")
                self._prefetch_q.put(result)  # blocks if queue is full (backpressure)
        finally:
            self._prefetch_q.put(None)  # sentinel always sent, even after an unexpected error

    def _consumer(self):
        while True:
            result = self._prefetch_q.get()

            if result is None:
                with self._lock:
                    self.status = "done"
                print(f"Session complete — cropped: {self.accepted}, skipped: {self.rejected}, no person/already done: {self.skipped}")
                return

            with self._lock:
                self.current = result
                self.status = "ready"

            choice = self._decision_q.get()

            with self._lock:
                if choice == "crop":
                    d = self.current
                    write_xmp(d["cr3_path"], d["x1"], d["y1"], d["x2"], d["y2"], d["w"], d["h"])
                    self.accepted += 1
                    print(f"  Cropped:   {d['cr3_path'].name}")
                else:
                    print(f"  Skipped:   {self.current['cr3_path'].name}")
                    self.rejected += 1
                self.current = None
                self.status = "loading"

    def decide(self, choice):
        """Called from Flask request handler. choice: 'crop' | 'skip'."""
        if self.status != "ready":
            return False
        self._decision_q.put(choice)
        return True

    def get_state(self):
        with self._lock:
            done_count = self.accepted + self.rejected
            if self.status == "done":
                return {
                    "status": "done",
                    "accepted": self.accepted,
                    "rejected": self.rejected,
                    "skipped": self.skipped,
                }
            buffered = self._prefetch_q.qsize()
            if self.status == "loading" or self.current is None:
                return {
                    "status": "loading",
                    "idx": done_count,
                    "total": self.total_eligible,
                    "buffered": buffered,
                }
            d = self.current
            return {
                "status": "ready",
                "filename": d["cr3_path"].name,
                "idx": done_count + 1,
                "total": self.total_eligible,
                "buffered": buffered,
                "orig_b64": base64.b64encode(d["orig_bytes"]).decode(),
                "crop_b64": base64.b64encode(d["crop_bytes"]).decode(),
            }


def create_app(initial_path: str = "", force: bool = False, all_people: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["review_state"] = None
    app.config["start_stage"] = None   # str while starting, None otherwise
    app.config["start_error"] = None   # str if start failed
    app.config["initial_path"] = initial_path
    app.config["force"] = force
    app.config["all_people"] = all_people

    @app.route("/")
    def index():
        return render_template("index.html", initial_path=app.config["initial_path"], prefetch=PREFETCH)

    @app.route("/api/pick-folder")
    def api_pick_folder():
        result = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select a folder of CR3 files")'],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return jsonify({"path": None})
        return jsonify({"path": result.stdout.strip()})

    @app.route("/api/start", methods=["POST"])
    def api_start():
        data = request.json
        path = Path(data.get("path", "")).expanduser().resolve()

        if not path.exists():
            return jsonify({"error": f"Path does not exist: {path}"}), 400

        app.config["review_state"] = None
        app.config["start_error"] = None
        app.config["start_stage"] = "Scanning folder..."

        def _do_start():
            try:
                print(f"Scanning {path} ...")
                files = [p for p in path.rglob("*") if p.suffix.lower() == ".cr3"]
                if not files:
                    app.config["start_error"] = "No CR3 files found in that folder"
                    print("  No CR3 files found.")
                    return

                n = len(files)
                print(f"  Found {n} CR3 files. Reading capture times...")
                app.config["start_stage"] = f"Reading capture times ({n} files)..."
                workers = min(8, n)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    times = list(pool.map(get_capture_time, files))

                empty = sum(1 for t in times if not t)
                if empty:
                    print(f"  Warning: {empty}/{n} files had no readable timestamp")
                else:
                    print(f"  All {n} timestamps read successfully")
                for name, t in [(f.name, t) for f, t in zip(files, times) if t][:3]:
                    print(f"    {name} → {t}")

                cr3_files = [
                    f for _, f in sorted(zip(times, files), key=lambda x: (x[0], str(x[1])))
                ]
                if cr3_files:
                    already_done = sum(1 for f in cr3_files if has_existing_crop(f))
                    print(f"  Sort order: {cr3_files[0].name} … {cr3_files[-1].name}")
                    print(f"  {already_done}/{n} already have crop data (will be skipped)")

                if not _models_ready.is_set():
                    print("  Waiting for AI model to finish loading...")
                    app.config["start_stage"] = "Loading AI model..."
                    _models_ready.wait()

                eligible = sum(1 for f in cr3_files if app.config["force"] or not has_existing_crop(f))
                print(f"  Starting review session: {eligible} photos to review")
                app.config["start_stage"] = f"Preparing {n} photos..."
                app.config["review_state"] = ReviewState(
                    cr3_files, _models,
                    force=app.config["force"],
                    all_people=app.config["all_people"],
                )
            except Exception as e:
                app.config["start_error"] = str(e)
                print(f"  Error during startup: {e}")
            finally:
                app.config["start_stage"] = None

        threading.Thread(target=_do_start, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/state")
    def api_state():
        error = app.config.get("start_error")
        if error:
            app.config["start_error"] = None
            return jsonify({"status": "error", "message": error})

        stage = app.config.get("start_stage")
        if stage is not None:
            return jsonify({"status": "starting", "stage": stage})

        state = app.config["review_state"]
        if state is None:
            return jsonify({"status": "waiting", "models_ready": _models_ready.is_set()})
        return jsonify(state.get_state())

    @app.route("/api/decide", methods=["POST"])
    def api_decide():
        state = app.config["review_state"]
        if state is None:
            return jsonify({"error": "no active session"}), 400
        choice = request.json.get("choice")
        if choice not in ("crop", "skip"):
            return jsonify({"error": "invalid choice"}), 400
        state.decide(choice)
        return jsonify({"ok": True})

    return app


def web_main():
    parser = argparse.ArgumentParser(
        description="Interactive web UI for reviewing and applying AI crops to CR3 files"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        type=str,
        help="Folder containing CR3 files (optional — can be selected in the browser)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Show files that already have crop data",
    )
    parser.add_argument(
        "-a", "--all-people",
        action="store_true",
        help="Crop to include all detected people",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the web server on (default: 5001)",
    )

    args = parser.parse_args()

    initial_path = ""
    if args.path:
        root = Path(args.path).expanduser().resolve()
        if not root.exists():
            raise SystemExit(f"Path does not exist: {root}")
        initial_path = str(root)

    threading.Thread(target=_load_models_thread, daemon=True).start()

    app = create_app(initial_path=initial_path, force=args.force, all_people=args.all_people)

    url = f"http://localhost:{args.port}"
    print(f"Starting review UI at {url} (use --port to change)")
    webbrowser.open(url)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    web_main()
