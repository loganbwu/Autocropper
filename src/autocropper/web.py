#!/usr/bin/env python3

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ANSI colour helpers — disabled automatically when output is not a terminal.
_C      = sys.stdout.isatty()
_GREEN  = "\033[32m" if _C else ""
_YELLOW = "\033[33m" if _C else ""
_RED    = "\033[31m" if _C else ""
_CYAN   = "\033[36m" if _C else ""
_DIM    = "\033[2m"  if _C else ""
_RESET  = "\033[0m"  if _C else ""

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .main import (
    compute_crop,
    has_been_reviewed,
    has_existing_crop,
    load_ml_models,
    load_models,
    recompute_crop,
    write_decline_marker,
    write_xmp,
)
from .ml_crop import TrainingDataset, predict_ml_crop

PREFETCH_DEFAULT = 10  # default prefetch buffer; user can override in the UI
PREFETCH_MIN = 1
PREFETCH_MAX = 200

MARGIN_DEFAULT = 0.20
MARGIN_MIN = 0.00
MARGIN_MAX = 0.50

# SSE: woken whenever any server-side state changes that the browser should see.
_sse_condition = threading.Condition()


def _notify_sse():
    with _sse_condition:
        _sse_condition.notify_all()


# Standard models load eagerly in background
_models = None
_models_ready = threading.Event()

# ML models (SAM2 + ViTPose) load lazily when first needed
_ml_models = None
_ml_models_ready = threading.Event()
_ml_models_loading = False
_ml_models_error: str = ""
_ml_models_lock = threading.Lock()


def _load_models_thread():
    global _models
    print("Loading AI model in background...")
    _models = load_models()
    print(f"{_GREEN}AI model ready.{_RESET}")
    _models_ready.set()


def _ensure_ml_models_loaded():
    """Trigger ML model loading if not already started; returns immediately."""
    global _ml_models, _ml_models_loading, _ml_models_error
    with _ml_models_lock:
        if _ml_models_ready.is_set() or _ml_models_loading:
            return
        _ml_models_loading = True
        _ml_models_error = ""

    def _load():
        global _ml_models, _ml_models_loading, _ml_models_error
        print("Loading ML models (SAM + ViTPose) in background...")
        try:
            _models_ready.wait()  # ensure base GDINO is loaded before we borrow it
            _ml_models = load_ml_models(gdino_models=_models)
            print(f"{_GREEN}ML models ready.{_RESET}")
            _ml_models_ready.set()
            _notify_sse()
        except Exception as e:
            print(f"{_RED}ERROR: failed to load ML models: {e}{_RESET}")
            import traceback; traceback.print_exc()
            with _ml_models_lock:
                _ml_models_error = str(e)
                _ml_models_loading = False  # allow retry
            _notify_sse()

    threading.Thread(target=_load, daemon=True).start()


class ReviewState:
    """Producer/consumer design: producer pre-processes up to PREFETCH images ahead,
    consumer presents them and waits for decisions. The user rarely waits."""

    def __init__(self, files, models, all_people=False, prefetch=PREFETCH_DEFAULT, pre_skipped=0,
                 ml_mode=False, ml_dataset=None):
        self.files = files
        self.models = models
        self.all_people = all_people
        self.prefetch = prefetch

        self.accepted = 0
        self.rejected = 0
        self.pre_skipped = pre_skipped       # already-reviewed files excluded before this session
        self.no_person_skipped = 0           # person not detected (or inference error)
        self.noop_skipped = 0                # person found but crop is ~full frame
        self.skipped = pre_skipped           # combined: pre + no_person + noop
        self.producer_processed = 0
        self.margin = MARGIN_DEFAULT
        self.status = "loading"
        self.current = None

        # ML mode state
        self.ml_mode = ml_mode
        self.ml_dataset = ml_dataset   # TrainingDataset | None

        self.total_eligible = len(files)

        # Only processed results enter the queue (skips handled inline by producer).
        # Unbounded queue; backpressure is handled via _prefetch_cv so the limit
        # can be changed live without recreating the queue.
        self._prefetch_q = queue.Queue()
        self._prefetch_cv = threading.Condition(threading.Lock())
        self._decision_q = queue.Queue()
        self._lock = threading.Lock()
        self._producer_gen = 0  # incremented on mode switch; old producer exits, new one starts
        self._decided_paths: set = set()  # CR3 paths the user has explicitly accepted or rejected

        threading.Thread(target=self._producer, args=(self.files, self._producer_gen), daemon=True).start()
        threading.Thread(target=self._consumer, daemon=True).start()

    def _producer(self, files, gen):
        """Worker thread. Exits early (without emitting sentinel) when _producer_gen changes."""

        try:
            for cr3 in files:
                if self._producer_gen != gen:
                    return

                with self._lock:
                    use_ml = self.ml_mode and self.ml_dataset is not None and _ml_models_ready.is_set()
                    margin = self.margin
                try:
                    if use_ml:
                        result = predict_ml_crop(cr3, self.ml_dataset, _ml_models)
                        if result is None:
                            print(f"{_YELLOW}  ML: no crop for {cr3.name}, falling back to classic crop{_RESET}")
                            result = compute_crop(self.models, cr3, self.all_people, margin_ratio=margin)
                    else:
                        result = compute_crop(self.models, cr3, self.all_people, margin_ratio=margin)
                except Exception as e:
                    print(f"{_RED}  Warning: skipping {cr3.name} — {e}{_RESET}")
                    with self._lock:
                        self.skipped += 1
                        self.no_person_skipped += 1
                        self.producer_processed += 1
                    _notify_sse()
                    continue

                with self._lock:
                    self.producer_processed += 1
                    if result is None:
                        self.skipped += 1
                        self.no_person_skipped += 1
                    elif result is False:
                        self.skipped += 1
                        self.noop_skipped += 1

                if result is None:
                    write_decline_marker(cr3)
                    _notify_sse()
                    print(f"{_YELLOW}  No person: {cr3.name}{_RESET}")
                    continue
                if result is False:
                    write_decline_marker(cr3)
                    _notify_sse()
                    print(f"{_YELLOW}  No-op crop: {cr3.name}{_RESET}")
                    continue

                print(f"{_GREEN}  Ready:     {cr3.name}{_RESET}")
                with self._prefetch_cv:
                    while self._prefetch_q.qsize() >= self.prefetch:
                        self._prefetch_cv.wait()
                self._prefetch_q.put(result)
                _notify_sse()
        finally:
            if self._producer_gen == gen:
                self._prefetch_q.put(None)

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
                _notify_sse()
                print(f"{_GREEN}Session complete — cropped: {self.accepted}, skipped: {self.rejected}, no person/already done: {self.skipped}{_RESET}")
                return

            with self._lock:
                if not result.get("ml_crop"):
                    recompute_crop(result, self.margin)
                self.current = result
                self.status = "ready"
            _notify_sse()

            decision = self._decision_q.get()

            # "_discard" is sent by api_ml_mode when the mode switches while an
            # image is displayed; just drop the current item and go back to waiting.
            choice, coords, angle = decision
            if choice == "_discard":
                continue  # coords: {x1,y1,x2,y2} or None; angle: degrees

            with self._lock:
                if choice == "crop":
                    d = self.current
                    x1 = coords["x1"] if coords else d["x1"]
                    y1 = coords["y1"] if coords else d["y1"]
                    x2 = coords["x2"] if coords else d["x2"]
                    y2 = coords["y2"] if coords else d["y2"]
                    method_kw = "AutoCropper_ML" if d.get("ml_crop") else "AutoCropper_Margin"
                    write_xmp(d["cr3_path"], x1, y1, x2, y2, d["w"], d["h"],
                              angle=angle or 0,
                              keywords=["AutoCropper", method_kw])
                    self.accepted += 1
                    self._decided_paths.add(d["cr3_path"])
                    print(f"{_GREEN}  Cropped:   {d['cr3_path'].name}{_RESET}")
                else:
                    d = self.current
                    write_decline_marker(d["cr3_path"])
                    self.rejected += 1
                    self._decided_paths.add(d["cr3_path"])
                    print(f"{_DIM}  Declined:  {d['cr3_path'].name}{_RESET}")
                self.current = None
                self.status = "loading"
            _notify_sse()

    def decide(self, choice, coords=None, angle=0):
        """Called from Flask request handler. choice: 'crop' | 'skip'.
        coords: optional {x1,y1,x2,y2} overriding server-side crop position.
        angle: crop rotation in degrees (CCW positive)."""
        if self.status != "ready":
            return False
        self._decision_q.put((choice, coords, angle))
        return True

    def get_state(self):
        with self._lock:
            done_count = self.accepted + self.rejected
            auto_skipped = self.no_person_skipped + self.noop_skipped
            # Include auto-skips in progress so the bar reflects true processing progress.
            reviewed_count = done_count + auto_skipped
            if self.status == "done":
                return {
                    "status": "done",
                    "accepted": self.accepted,
                    "rejected": self.rejected,
                    "pre_skipped": self.pre_skipped,
                    "no_person_skipped": self.no_person_skipped,
                    "noop_skipped": self.noop_skipped,
                }
            buffered = self._prefetch_q.qsize()
            # Peek at queue contents (internal deque) to get per-slot ML flag.
            buffer_types = [bool(item.get("ml_crop")) for item in list(self._prefetch_q.queue)
                            if item is not None]
            if self.status == "loading" or self.current is None:
                return {
                    "status": "loading",
                    "idx": reviewed_count,
                    "producer_idx": self.producer_processed,
                    "total": self.total_eligible,
                    "buffered": buffered,
                    "buffer_types": buffer_types,
                    "prefetch": self.prefetch,
                    "ml_mode": self.ml_mode,
                    "ml_dataset_loaded": self.ml_dataset is not None,
                }
            d = self.current
            state = {
                "status": "ready",
                "filename": d["cr3_path"].name,
                "idx": reviewed_count + 1,
                "total": self.total_eligible,
                "buffered": buffered,
                "buffer_types": buffer_types,
                "prefetch": self.prefetch,
                "orig_b64": base64.b64encode(d["orig_bytes"]).decode(),
                "crop_coords": {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]},
                "ml_crop": bool(d.get("ml_crop")),
                "ml_mode": self.ml_mode,
                "ml_dataset_loaded": self.ml_dataset is not None,
                "ml_models_ready": _ml_models_ready.is_set(),
            }
            return state


def create_app(initial_path: str = "", force: bool = False, all_people: bool = False,
               ml_dataset_path: str = "") -> Flask:
    app = Flask(__name__)
    app.config["review_state"] = None
    app.config["start_stage"] = None     # str while starting, None otherwise
    app.config["start_progress"] = None  # float 0-1 during linear stages, None otherwise
    app.config["start_error"] = None     # str if start failed
    app.config["initial_path"] = initial_path
    app.config["force"] = force
    app.config["all_people"] = all_people
    app.config["ml_dataset"] = None      # TrainingDataset | None, persists across sessions
    app.config["ml_dataset_name"] = ""   # filename stem shown in the UI
    app.config["ml_dataset_records"] = 0

    if ml_dataset_path:
        try:
            ds = TrainingDataset.load(ml_dataset_path)
            app.config["ml_dataset"] = ds
            app.config["ml_dataset_name"] = Path(ml_dataset_path).name
            app.config["ml_dataset_records"] = len(ds.records)
            print(f"Loaded ML dataset: {len(ds.records)} records from {ml_dataset_path}")
            _ensure_ml_models_loaded()
        except Exception as e:
            print(f"Warning: could not load ML dataset from {ml_dataset_path}: {e}")

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
                    print(f"{_RED}  No CR3 files found.{_RESET}")
                    _notify_sse()
                    return

                n = len(files)
                print(f"  Found {n} CR3 files. Checking review status...")
                app.config["start_stage"] = f"Scanning {n} files..."
                _notify_sse()

                # Use mtime for ordering (instant stat, no 12 MB header reads).
                # has_been_reviewed reads a small XMP sidecar — keep in parallel.
                reviewed_set = set()
                with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
                    futures = {pool.submit(has_been_reviewed, f): f for f in files}
                    for future in as_completed(futures):
                        f = futures[future]
                        try:
                            if future.result():
                                reviewed_set.add(f)
                        except Exception:
                            pass

                cr3_files = sorted(files, key=lambda f: (f.stat().st_mtime, f.name))

                force = app.config["force"]
                already_reviewed = [f for f in cr3_files if not force and f in reviewed_set]
                to_review = [f for f in cr3_files if force or f not in reviewed_set]

                if cr3_files:
                    print(f"  Sort order: {cr3_files[0].name} … {cr3_files[-1].name}")
                    print(f"  {len(already_reviewed)}/{n} already reviewed (will be skipped)")

                if not _models_ready.is_set():
                    print("  Waiting for AI model to finish loading...")
                    app.config["start_stage"] = "Loading AI model..."
                    _notify_sse()
                    _models_ready.wait()

                dataset = app.config.get("ml_dataset")
                if dataset is not None and not _ml_models_ready.is_set() and _ml_models_loading:
                    print("  Waiting for ML models to finish loading...")
                    app.config["start_stage"] = "Loading ML models..."
                    _notify_sse()
                    _ml_models_ready.wait(timeout=120)
                use_ml = dataset is not None and _ml_models_ready.is_set()
                mode_str = "ML" if use_ml else "classic"
                print(f"  Starting review session: {len(to_review)} photos to review, mode={mode_str}, prefetch={PREFETCH_DEFAULT}")
                app.config["start_stage"] = f"Preparing {n} photos..."
                _notify_sse()
                app.config["review_state"] = ReviewState(
                    to_review, _models,
                    pre_skipped=len(already_reviewed),
                    all_people=app.config["all_people"],
                    ml_mode=use_ml,
                    ml_dataset=dataset if use_ml else None,
                )
            except Exception as e:
                app.config["start_error"] = str(e)
                print(f"{_RED}  Error during startup: {e}{_RESET}")
            finally:
                app.config["start_stage"] = None
                _notify_sse()

        threading.Thread(target=_do_start, daemon=True).start()
        return jsonify({"ok": True})

    def _ml_info():
        return {
            "ml_dataset_loaded": app.config["ml_dataset"] is not None,
            "ml_dataset_name": app.config["ml_dataset_name"],
            "ml_dataset_records": app.config["ml_dataset_records"],
            "ml_models_ready": _ml_models_ready.is_set(),
            "ml_models_error": _ml_models_error,
        }

    @app.route("/api/state")
    def api_state():
        error = app.config.get("start_error")
        if error:
            app.config["start_error"] = None
            return jsonify({"status": "error", "message": error})

        stage = app.config.get("start_stage")
        if stage is not None:
            return jsonify({"status": "starting", "stage": stage,
                            "progress": app.config.get("start_progress"),
                            **_ml_info()})

        state = app.config["review_state"]
        if state is None:
            return jsonify({"status": "waiting", "models_ready": _models_ready.is_set(),
                            **_ml_info()})
        return jsonify({**state.get_state(), **_ml_info()})

    @app.route("/api/decide", methods=["POST"])
    def api_decide():
        state = app.config["review_state"]
        if state is None:
            return jsonify({"error": "no active session"}), 400
        choice = request.json.get("choice")
        if choice not in ("crop", "skip"):
            return jsonify({"error": "invalid choice"}), 400
        coords = request.json.get("coords")  # {x1,y1,x2,y2} or absent
        angle  = request.json.get("angle", 0)
        state.decide(choice, coords, angle)
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
            if not d.get("ml_crop"):
                recompute_crop(d, margin)
            crop_coords = {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]}
        return jsonify({"ok": True, "crop_coords": crop_coords})

    @app.route("/api/ml-dataset", methods=["POST"])
    def api_ml_dataset():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        f = request.files["file"]
        try:
            dataset = TrainingDataset.load(f.stream)
        except Exception as e:
            return jsonify({"error": f"could not load dataset: {e}"}), 400

        _ensure_ml_models_loaded()

        state = app.config["review_state"]
        if state is not None:
            with state._lock:
                state.ml_dataset = dataset
        app.config["ml_dataset"] = dataset
        app.config["ml_dataset_name"] = f.filename or "dataset.pkl"
        app.config["ml_dataset_records"] = len(dataset.records)
        return jsonify({
            "ok": True,
            "records": len(dataset.records),
            "alpha": dataset.alpha,
            "ml_models_ready": _ml_models_ready.is_set(),
        })

    @app.route("/api/ml-mode", methods=["POST"])
    def api_ml_mode():
        data = request.json or {}
        enabled = bool(data.get("enabled", False))

        state = app.config["review_state"]
        dataset = app.config.get("ml_dataset")

        if enabled and dataset is None:
            return jsonify({"error": "upload a training dataset first"}), 400
        if enabled and not _ml_models_ready.is_set():
            return jsonify({"error": "ML models are still loading, please wait"}), 400

        if state is not None:
            with state._lock:
                old_mode = state.ml_mode
                state.ml_mode = enabled
                state.ml_dataset = dataset if enabled else None
                current_item = state.current
                if old_mode != enabled:
                    state._producer_gen += 1
                    new_gen = state._producer_gen
                    state.current = None
                    state.status = "loading"
                    # Reset per-pass auto-skip counters so the new producer starts clean.
                    state.skipped = state.pre_skipped
                    state.no_person_skipped = 0
                    state.noop_skipped = 0
                    state.producer_processed = len(state._decided_paths)

            if old_mode != enabled:
                # If an image was on screen, unblock the consumer so it stops waiting
                # for a user decision and goes back to reading from the queue.
                if current_item is not None:
                    state._decision_q.put(("_discard", None, 0))

                # Drain the buffer entirely; the new producer will repopulate it.
                while True:
                    try:
                        state._prefetch_q.get_nowait()
                    except queue.Empty:
                        break

                # Wake up the old producer if it was blocked waiting for buffer space.
                with state._prefetch_cv:
                    state._prefetch_cv.notify_all()

                # Start fresh from photo #1 of files the user has not yet decided on.
                remaining = [f for f in state.files if f not in state._decided_paths]
                threading.Thread(
                    target=state._producer,
                    args=(remaining, new_gen),
                    daemon=True,
                ).start()
                _notify_sse()

        return jsonify({"ok": True, "ml_mode": enabled})

    @app.route("/api/events")
    def api_events():
        def get_snapshot():
            error = app.config.get("start_error")
            if error:
                app.config["start_error"] = None
                return {"status": "error", "message": error}
            stage = app.config.get("start_stage")
            if stage is not None:
                return {"status": "starting", "stage": stage,
                        "progress": app.config.get("start_progress"),
                        **_ml_info()}
            state = app.config["review_state"]
            if state is None:
                return {"status": "waiting", "models_ready": _models_ready.is_set(),
                        **_ml_info()}
            return {**state.get_state(), **_ml_info()}

        def generate():
            yield f"data: {json.dumps(get_snapshot())}\n\n"
            while True:
                with _sse_condition:
                    _sse_condition.wait(timeout=25)
                yield f"data: {json.dumps(get_snapshot())}\n\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
    parser.add_argument(
        "--ml-dataset", "-m",
        type=str,
        default="",
        metavar="DATASET",
        help="Path to a training dataset .pkl file to pre-load for ML crop mode",
    )

    args = parser.parse_args()

    initial_path = ""
    if args.path:
        root = Path(args.path).expanduser().resolve()
        if not root.exists():
            raise SystemExit(f"Path does not exist: {root}")
        initial_path = str(root)

    ml_dataset_path = ""
    if args.ml_dataset:
        p = Path(args.ml_dataset).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ML dataset not found: {p}")
        ml_dataset_path = str(p)
    else:
        candidates = sorted(Path("data").glob("*.pkl")) if Path("data").is_dir() else []
        if candidates:
            ml_dataset_path = str(candidates[0])
            print(f"{_CYAN}Auto-detected ML dataset: {ml_dataset_path}{_RESET}")

    threading.Thread(target=_load_models_thread, daemon=True).start()

    app = create_app(initial_path=initial_path, force=args.force, all_people=args.all_people,
                     ml_dataset_path=ml_dataset_path)

    url = f"http://localhost:{args.port}"
    print(f"{_GREEN}Starting review UI at {url}{_RESET} (use --port to change)")
    webbrowser.open(url)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    web_main()
