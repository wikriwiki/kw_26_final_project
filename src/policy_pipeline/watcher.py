from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.policy_pipeline.state import PolicyStatus, append_status_for_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WATCH_DIR = PROJECT_ROOT / "data" / "policies" / "inbox"
SUPPORTED_SUFFIXES = {".json", ".txt"}
STABILITY_CHECK_INTERVAL_SECONDS = 1.0
STABILITY_REQUIRED_CHECKS = 2
EVENT_DEBOUNCE_SECONDS = 2.0


def is_supported_policy_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def wait_until_file_stable(
    path: Path,
    interval_seconds: float = STABILITY_CHECK_INTERVAL_SECONDS,
    required_checks: int = STABILITY_REQUIRED_CHECKS,
) -> bool:
    stable_checks = 0
    previous_size: int | None = None

    while stable_checks < required_checks:
        if not path.exists() or not path.is_file():
            return False

        current_size = path.stat().st_size

        if previous_size is not None and current_size == previous_size:
            stable_checks += 1
        else:
            stable_checks = 0

        previous_size = current_size
        time.sleep(interval_seconds)

    return True


class PolicyFileEventHandler(FileSystemEventHandler):
    """Detect policy text files and enqueue each stable file once."""

    def __init__(self, processing_queue: queue.Queue[Path]) -> None:
        self.processing_queue = processing_queue
        self._last_event_at: dict[Path, float] = {}
        self._queued_paths: set[Path] = set()
        self._lock = threading.Lock()

    def _handle_path(self, raw_path: str, event_type: str) -> None:
        path = Path(raw_path).resolve()

        if not is_supported_policy_file(path):
            return

        now = time.monotonic()
        with self._lock:
            last_event_at = self._last_event_at.get(path, 0.0)
            if now - last_event_at < EVENT_DEBOUNCE_SECONDS:
                print(f"[policy-watch] duplicate event ignored: {path}")
                return

            self._last_event_at[path] = now

        print(f"[policy-watch] {event_type}: {path}")
        threading.Thread(
            target=self._enqueue_when_stable,
            args=(path,),
            daemon=True,
        ).start()

    def _enqueue_when_stable(self, path: Path) -> None:
        if not wait_until_file_stable(path):
            print(f"[policy-watch] file disappeared before stable: {path}")
            return

        with self._lock:
            if path in self._queued_paths:
                print(f"[policy-watch] already queued: {path}")
                return

            self._queued_paths.add(path)

        record = append_status_for_file(path, PolicyStatus.DETECTED)
        self.processing_queue.put(path)
        print(f"[policy-watch] queued stable file: {path} ({record.policy_id})")

    def on_created(self, event) -> None:
        self._handle_path(event.src_path, "created")

    def on_modified(self, event) -> None:
        self._handle_path(event.src_path, "modified")


def watch_policy_inbox(watch_dir: Path = DEFAULT_WATCH_DIR) -> None:
    watch_dir.mkdir(parents=True, exist_ok=True)

    processing_queue: queue.Queue[Path] = queue.Queue()
    event_handler = PolicyFileEventHandler(processing_queue)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    worker_thread = threading.Thread(
        target=process_policy_queue,
        args=(processing_queue,),
        daemon=True,
    )
    worker_thread.start()

    print(f"[policy-watch] watching: {watch_dir}")
    supported_suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
    print(f"[policy-watch] supported file types: {supported_suffixes}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[policy-watch] stopping")
        observer.stop()

    observer.join()


def process_policy_queue(processing_queue: queue.Queue[Path]) -> None:
    while True:
        path = processing_queue.get()
        try:
            print(f"[policy-watch] processing ready: {path}")
        finally:
            processing_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch policy inbox for .txt files.")
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=DEFAULT_WATCH_DIR,
        help="Directory to watch for policy text files.",
    )
    args = parser.parse_args()

    watch_policy_inbox(args.watch_dir)


if __name__ == "__main__":
    main()
