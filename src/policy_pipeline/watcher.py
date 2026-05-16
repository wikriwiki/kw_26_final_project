"""
watcher.py
==========
data/policies/inbox/ 감시 → 안정화된 파일을 큐로 enqueue → 워커 스레드가
`pipeline.process_policy_file()` 호출.

변경 사항 (v2):
- 워커가 실제로 pipeline 을 호출하게 됨 (이전엔 큐에서 꺼내고 로그만 찍었음).
- 큐 dedup: 같은 path 가 이미 큐에 있거나 처리 중이면 skip.
- 임시 파일 / 점 파일 / 지원 외 확장자 무시.
- LLM 클라이언트는 첫 호출 시 lazy 초기화 (테스트에서 watcher import 만 해도
  OPENAI_API_KEY 가 필요해지는 부작용 차단).
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.policy_pipeline.pipeline import process_policy_file
from src.policy_pipeline.state import PolicyStatus, append_status_for_file


log = logging.getLogger("policy_watch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WATCH_DIR = PROJECT_ROOT / "data" / "policies" / "inbox"
SUPPORTED_SUFFIXES = {".json", ".txt"}
STABILITY_CHECK_INTERVAL_SECONDS = 1.0
STABILITY_REQUIRED_CHECKS = 2
EVENT_DEBOUNCE_SECONDS = 2.0


# ---------------------------------------------------------------------------
# 파일 필터
# ---------------------------------------------------------------------------
def is_supported_policy_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    name = path.name
    # 흔한 임시 파일 패턴 무시
    if name.startswith(".") or name.startswith("~") or name.endswith(".tmp"):
        return False
    if name.endswith(".swp") or name.endswith(".part") or name.endswith(".crdownload"):
        return False
    return True


def wait_until_file_stable(
    path: Path,
    interval_seconds: float = STABILITY_CHECK_INTERVAL_SECONDS,
    required_checks: int = STABILITY_REQUIRED_CHECKS,
) -> bool:
    """파일 크기가 N회 연속 동일하면 안정으로 판단."""
    stable = 0
    previous: int | None = None

    while stable < required_checks:
        if not path.exists() or not path.is_file():
            return False
        current = path.stat().st_size
        if previous is not None and current == previous:
            stable += 1
        else:
            stable = 0
        previous = current
        time.sleep(interval_seconds)
    return True


# ---------------------------------------------------------------------------
# 이벤트 핸들러 + 큐
# ---------------------------------------------------------------------------
class PolicyFileEventHandler(FileSystemEventHandler):
    """파일 생성/수정 이벤트 → 안정화 확인 → 큐 enqueue."""

    def __init__(self, processing_queue: queue.Queue[Path]) -> None:
        self.processing_queue = processing_queue
        self._last_event_at: dict[Path, float] = {}
        self._queued_or_inflight: set[Path] = set()
        self._lock = threading.Lock()

    # 외부 인터페이스 - 워커가 처리 끝나면 호출해서 inflight 해제
    def mark_done(self, path: Path) -> None:
        with self._lock:
            self._queued_or_inflight.discard(path)

    def _handle_path(self, raw_path: str, event_type: str) -> None:
        path = Path(raw_path).resolve()
        if not is_supported_policy_file(path):
            return

        now = time.monotonic()
        with self._lock:
            last = self._last_event_at.get(path, 0.0)
            if now - last < EVENT_DEBOUNCE_SECONDS:
                return
            self._last_event_at[path] = now
            if path in self._queued_or_inflight:
                return  # 이미 큐에 있거나 처리 중

        log.info("[watch] %s: %s", event_type, path)
        threading.Thread(
            target=self._enqueue_when_stable,
            args=(path,),
            daemon=True,
        ).start()

    def _enqueue_when_stable(self, path: Path) -> None:
        if not wait_until_file_stable(path):
            log.warning("[watch] file disappeared before stable: %s", path)
            return

        with self._lock:
            if path in self._queued_or_inflight:
                return
            self._queued_or_inflight.add(path)

        append_status_for_file(path, PolicyStatus.DETECTED)
        self.processing_queue.put(path)
        log.info("[watch] queued: %s", path)

    def on_created(self, event) -> None:
        self._handle_path(event.src_path, "created")

    def on_modified(self, event) -> None:
        self._handle_path(event.src_path, "modified")


# ---------------------------------------------------------------------------
# 워커
# ---------------------------------------------------------------------------
def _make_default_llm_client():
    # 지연 import — `python -m src.policy_pipeline.watcher --help` 만으로는
    # API 키 환경변수가 필요 없게.
    from src.infra.llm.openai_client import OpenAIChatClient
    return OpenAIChatClient()


def process_policy_queue(
    processing_queue: queue.Queue[Path],
    handler: PolicyFileEventHandler,
    llm_client_factory=_make_default_llm_client,
) -> None:
    """단일 워커 스레드. 큐에서 path 꺼내 pipeline 호출.

    LLM 클라이언트는 한 번만 만들어 재사용.
    """
    llm_client = None
    while True:
        path = processing_queue.get()
        try:
            if llm_client is None:
                llm_client = llm_client_factory()
            result = process_policy_file(path, llm_client)
            log.info(
                "[watch] processed %s → %s (keys=%d jobs=%d %s)",
                path.name,
                result.final_status.value,
                result.cache_keys_invalidated,
                result.summary_jobs_enqueued,
                "[duplicate]" if result.skipped_as_duplicate else "",
            )
        except Exception:  # noqa
            log.exception("[watch] worker crashed processing %s", path)
        finally:
            handler.mark_done(path)
            processing_queue.task_done()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def watch_policy_inbox(
    watch_dir: Path = DEFAULT_WATCH_DIR,
    *,
    use_polling: bool = False,
) -> None:
    """inbox 감시 시작. 무한 루프. KeyboardInterrupt 로 종료.

    `use_polling=True` 면 inotify 가 동작 안하는 환경(WSL, 네트워크 마운트)에서
    polling observer 로 폴백.
    """
    watch_dir.mkdir(parents=True, exist_ok=True)

    processing_queue: queue.Queue[Path] = queue.Queue()
    handler = PolicyFileEventHandler(processing_queue)

    if use_polling:
        from watchdog.observers.polling import PollingObserver
        observer = PollingObserver()
    else:
        observer = Observer()

    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    worker = threading.Thread(
        target=process_policy_queue,
        args=(processing_queue, handler),
        daemon=True,
        name="policy-worker",
    )
    worker.start()

    log.info("[watch] watching: %s (polling=%s)", watch_dir, use_polling)
    log.info("[watch] supported: %s", ", ".join(sorted(SUPPORTED_SUFFIXES)))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("[watch] stopping")
        observer.stop()

    observer.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch policy inbox and run pipeline.")
    parser.add_argument(
        "--watch-dir", type=Path, default=DEFAULT_WATCH_DIR,
        help="Directory to watch for policy text files.",
    )
    parser.add_argument(
        "--polling", action="store_true",
        help="Use PollingObserver (for WSL / network mounts).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    watch_policy_inbox(args.watch_dir, use_polling=args.polling)


if __name__ == "__main__":
    main()
