#!/usr/bin/env python3

import argparse
import base64
import queue
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .main import (
    compute_crop,
    find_cr3_files,
    has_existing_crop,
    load_models,
    write_xmp,
)

DEFAULT_ROOT = Path.home() / "Desktop/Test"


class ReviewState:
    """Manages the queue of images and coordinates background processing."""

    def __init__(self, files, models, force=False, all_people=False):
        self.files = files
        self.models = models
        self.force = force
        self.all_people = all_people

        self.idx = 0
        self.accepted = 0
        self.rejected = 0
        self.skipped = 0

        self.status = "loading"  # "loading" | "ready" | "done"
        self.current = None      # dict from compute_crop

        self._decision_queue = queue.Queue()
        self._lock = threading.Lock()

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.idx < len(self.files):
            cr3 = self.files[self.idx]

            if not self.force and has_existing_crop(cr3):
                with self._lock:
                    self.idx += 1
                    self.skipped += 1
                continue

            with self._lock:
                self.status = "loading"
                self.current = None

            result = compute_crop(self.models, cr3, self.all_people)

            if result is None:
                with self._lock:
                    self.idx += 1
                    self.skipped += 1
                continue

            with self._lock:
                self.current = result
                self.status = "ready"

            # Block until user makes a decision
            choice = self._decision_queue.get()

            with self._lock:
                if choice == "crop":
                    d = self.current
                    write_xmp(d["cr3_path"], d["x1"], d["y1"], d["x2"], d["y2"], d["w"], d["h"])
                    self.accepted += 1
                else:
                    self.rejected += 1
                self.idx += 1
                self.current = None
                self.status = "loading"

        with self._lock:
            self.status = "done"

    def decide(self, choice):
        """Called from Flask request handler. choice: 'crop' | 'skip'."""
        if self.status != "ready":
            return False
        self._decision_queue.put(choice)
        return True

    def get_state(self):
        with self._lock:
            eligible = len(self.files) - self.skipped
            done_count = self.accepted + self.rejected
            if self.status == "done":
                return {
                    "status": "done",
                    "accepted": self.accepted,
                    "rejected": self.rejected,
                    "skipped": self.skipped,
                }
            if self.status == "loading" or self.current is None:
                return {
                    "status": "loading",
                    "idx": done_count,
                    "total": eligible,
                }
            d = self.current
            return {
                "status": "ready",
                "filename": d["cr3_path"].name,
                "idx": done_count + 1,
                "total": eligible,
                "orig_b64": base64.b64encode(d["orig_bytes"]).decode(),
                "crop_b64": base64.b64encode(d["crop_bytes"]).decode(),
            }


def create_app(state: ReviewState) -> Flask:
    app = Flask(__name__)
    app.config["state"] = state

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/state")
    def api_state():
        return jsonify(app.config["state"].get_state())

    @app.route("/api/decide", methods=["POST"])
    def api_decide():
        choice = request.json.get("choice")
        if choice not in ("crop", "skip"):
            return jsonify({"error": "invalid choice"}), 400
        app.config["state"].decide(choice)
        return jsonify({"ok": True})

    return app


def web_main():
    parser = argparse.ArgumentParser(
        description="Interactive web UI for reviewing and applying AI crops to CR3 files"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_ROOT,
        type=Path,
        help="Folder containing CR3 files (default: ~/Desktop/Test)",
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
        default=5000,
        help="Port to run the web server on (default: 5000)",
    )

    args = parser.parse_args()
    root = args.path.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    cr3_files = find_cr3_files(root)
    if not cr3_files:
        raise SystemExit(f"No CR3 files found under: {root}")

    print(f"Found {len(cr3_files)} CR3 file(s). Loading models...")
    models = load_models()

    state = ReviewState(cr3_files, models, force=args.force, all_people=args.all_people)
    app = create_app(state)

    url = f"http://localhost:{args.port}"
    print(f"Starting review UI at {url}")
    webbrowser.open(url)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    web_main()
