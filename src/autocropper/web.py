#!/usr/bin/env python3

import argparse
import base64
import collections
import itertools
import queue
import subprocess
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .main import (
    compute_crop,
    get_capture_time,
    has_been_reviewed,
    has_existing_crop,
    load_models,
    recompute_crop,
    write_decline_marker,
    write_xmp,
)

PREFETCH_DEFAULT = 10  # default prefetch buffer; user can override in the UI
PREFETCH_MIN = 1
PREFETCH_MAX = 100
PROCESSING_WORKERS = 4  # parallel exiftool + file-read threads; inference is still serialised

MARGIN_DEFAULT = 0.20
MARGIN_MIN = 0.00
MARGIN_MAX = 0.50

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

    def __init__(self, files, models, all_people=False, prefetch=PREFETCH_DEFAULT, pre_skipped=0):
        self.files = files
        self.models = models
        self.all_people = all_people
        self.prefetch = prefetch

        self.accepted = 0
        self.rejected = 0
        self.skipped = pre_skipped
        self.producer_processed = 0
        self.margin = MARGIN_DEFAULT
        self.status = "loading"
        self.current = None

        self.total_eligible = len(files)

        # Only processed results enter the queue (skips handled inline by producer).
        # Unbounded queue; backpressure is handled via _prefetch_cv so the limit
        # can be changed live without recreating the queue.
        self._prefetch_q = queue.Queue()
        self._prefetch_cv = threading.Condition(threading.Lock())
        self._decision_q = queue.Queue()
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()  # serialises GPU/MPS model calls across worker threads

        threading.Thread(target=self._producer, daemon=True).start()
        threading.Thread(target=self._consumer, daemon=True).start()

    def _producer(self):
        def _process(cr3):
            return compute_crop(self.models, cr3, self.all_people,
                                _inference_lock=self._inference_lock, margin_ratio=self.margin)

        try:
            # Sliding-window thread pool: PROCESSING_WORKERS threads run file I/O in
            # parallel; the _inference_lock inside compute_crop serialises the GPU/MPS
            # model call so at most one inference runs at a time. Results are collected
            # in submission order so the review queue is ordered.
            with ThreadPoolExecutor(max_workers=PROCESSING_WORKERS) as pool:
                pending = collections.deque()
                files_iter = iter(self.files)

                for cr3 in itertools.islice(files_iter, PROCESSING_WORKERS):
                    pending.append((cr3, pool.submit(_process, cr3)))

                while pending:
                    cr3, future = pending.popleft()

                    # Keep the window full: submit next file while we wait for this result.
                    try:
                        next_cr3 = next(files_iter)
                        pending.append((next_cr3, pool.submit(_process, next_cr3)))
                    except StopIteration:
                        pass

                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  Warning: skipping {cr3.name} — {e}")
                        with self._lock:
                            self.skipped += 1
                            self.producer_processed += 1
                        continue

                    with self._lock:
                        self.producer_processed += 1
                        if result is None:
                            self.skipped += 1

                    if result is None:
                        print(f"  No person: {cr3.name}")
                        continue

                    print(f"  Ready:     {cr3.name}")
                    with self._prefetch_cv:
                        while self._prefetch_q.qsize() >= self.prefetch:
                            self._prefetch_cv.wait()
                    self._prefetch_q.put(result)
        finally:
            self._prefetch_q.put(None)  # sentinel always sent, even after an unexpected error

    def set_prefetch(self, n):
        with self._lock:
            self.prefetch = n
        with self._prefetch_cv:
            self._prefetch_cv.notify_all()  # wake producer if it was waiting under old limit
        print(f"  Prefetch buffer resized to {n}")

    def _consumer(self):
        while True:
            result = self._prefetch_q.get()
            with self._prefetch_cv:
                self._prefetch_cv.notify()  # one slot freed; let producer continue

            if result is None:
                with self._lock:
                    self.status = "done"
                print(f"Session complete — cropped: {self.accepted}, skipped: {self.rejected}, no person/already done: {self.skipped}")
                return

            with self._lock:
                recompute_crop(result, self.margin)
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
                    d = self.current
                    write_decline_marker(d["cr3_path"])
                    print(f"  Declined:  {d['cr3_path'].name}")
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
                    "producer_idx": self.producer_processed,
                    "total": self.total_eligible,
                    "buffered": buffered,
                    "prefetch": self.prefetch,
                }
            d = self.current
            return {
                "status": "ready",
                "filename": d["cr3_path"].name,
                "idx": done_count + 1,
                "total": self.total_eligible,
                "buffered": buffered,
                "prefetch": self.prefetch,
                "orig_b64": base64.b64encode(d["orig_bytes"]).decode(),
                "crop_b64": base64.b64encode(d["crop_bytes"]).decode(),
            }


def create_app(initial_path: str = "", force: bool = False, all_people: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["review_state"] = None
    app.config["start_stage"] = None     # str while starting, None otherwise
    app.config["start_progress"] = None  # float 0-1 during linear stages, None otherwise
    app.config["start_error"] = None     # str if start failed
    app.config["initial_path"] = initial_path
    app.config["force"] = force
    app.config["all_people"] = all_people

    @app.route("/")
    def index():
        return render_template("index.html", initial_path=app.config["initial_path"],
                               prefetch_default=PREFETCH_DEFAULT, prefetch_min=PREFETCH_MIN, prefetch_max=PREFETCH_MAX,
                               margin_default=int(MARGIN_DEFAULT * 100), margin_min=int(MARGIN_MIN * 100), margin_max=int(MARGIN_MAX * 100))

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
                print(f"  Found {n} CR3 files. Scanning...")
                app.config["start_stage"] = f"Scanning {n} files..."
                app.config["start_progress"] = 0.0
                workers = min(8, n)
                times_dict = {}
                reviewed_set = set()

                def _read_info(f):
                    return get_capture_time(f), has_been_reviewed(f)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_read_info, f): f for f in files}
                    for i, future in enumerate(as_completed(futures), 1):
                        f = futures[future]
                        try:
                            t, reviewed = future.result()
                        except Exception:
                            t, reviewed = '', False
                        times_dict[f] = t
                        if reviewed:
                            reviewed_set.add(f)
                        app.config["start_progress"] = i / n
                times = [times_dict[f] for f in files]
                app.config["start_progress"] = None

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

                force = app.config["force"]
                already_reviewed = [f for f in cr3_files if not force and f in reviewed_set]
                to_review = [f for f in cr3_files if force or f not in reviewed_set]

                if cr3_files:
                    print(f"  Sort order: {cr3_files[0].name} … {cr3_files[-1].name}")
                    print(f"  {len(already_reviewed)}/{n} already reviewed (will be skipped)")

                if not _models_ready.is_set():
                    print("  Waiting for AI model to finish loading...")
                    app.config["start_stage"] = "Loading AI model..."
                    _models_ready.wait()

                print(f"  Starting review session: {len(to_review)} photos to review, prefetch={PREFETCH_DEFAULT}")
                app.config["start_stage"] = f"Preparing {n} photos..."
                app.config["review_state"] = ReviewState(
                    to_review, _models,
                    pre_skipped=len(already_reviewed),
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
            return jsonify({"status": "starting", "stage": stage,
                            "progress": app.config.get("start_progress")})

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

    @app.route("/api/set-margin", methods=["POST"])
    def api_set_margin():
        state = app.config["review_state"]
        if state is None:
            return jsonify({"error": "no active session"}), 400
        pct = request.json.get("margin")
        if not isinstance(pct, (int, float)) or not (MARGIN_MIN * 100 <= pct <= MARGIN_MAX * 100):
            return jsonify({"error": "invalid value"}), 400
        margin = pct / 100.0
        with state._lock:
            state.margin = margin
            d = state.current
            if d is None:
                return jsonify({"ok": True})
            recompute_crop(d, margin)
            crop_b64 = base64.b64encode(d["crop_bytes"]).decode()
        return jsonify({"ok": True, "crop_b64": crop_b64})

    @app.route("/api/set-prefetch", methods=["POST"])
    def api_set_prefetch():
        state = app.config["review_state"]
        if state is None:
            return jsonify({"error": "no active session"}), 400
        n = request.json.get("prefetch")
        if not isinstance(n, int) or not (PREFETCH_MIN <= n <= PREFETCH_MAX):
            return jsonify({"error": "invalid value"}), 400
        state.set_prefetch(n)
        return jsonify({"ok": True, "prefetch": n})

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
