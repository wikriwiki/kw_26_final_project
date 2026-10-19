def run_day(agents: list[str], today: date, day_idx: int, workers: int = 64) -> dict:
    if not agents or any(not isinstance(a, str) or not a for a in agents) or len(set(agents)) != len(agents):
        raise ValueError("cohort must contain distinct nonempty agent IDs")
    day_str = today.isoformat()
    cohort = {"run_id": os.environ.get("SIM_RUN_ID") or str(OUT_DIR.resolve()),
              "day": day_str, "agent_ids": sorted(agents),
              "execution_fingerprint": execution_fingerprint()}
    cohort_path = OUT_DIR / f"cohort_{day_str}.json"
    if cohort_path.exists() and json.loads(cohort_path.read_text(encoding="utf-8")) != cohort:
        raise ValueError("cohort or execution settings changed; refusing to resume")
    atomic_json(cohort_path, cohort)
    done_path = CHECK_DIR / f"done_{day_str}.json"
    failed_path = CHECK_DIR / f"failed_{day_str}.json"
    metrics_path = METRICS_DIR / f"day_{day_str}.jsonl"

    # The sealed metrics file is the completed-agent ledger. Failure attempts
    # live separately, so a retry never leaves duplicate/error rows in the
    # auditable day. The transactional outbox is checked by process_one.
    done_aids: set[str] = set()
    skipped_aids: set[str] = set()
    ok_count = 0
    if metrics_path.exists():
        from day_resume import read_metric_rows
        for row in read_metric_rows(metrics_path):
            if (row.get("status") not in {"ok", "skipped"} or row.get("aid") not in agents
                    or row["aid"] in done_aids):
                raise ValueError("Existing daily metrics contain invalid, foreign or duplicate rows")
            if configured_context():
                from evidence_integrity import verify
                verify(row)
                if (row.get("experience_day") != day_str
                        or row.get("experience_run_id") != cohort["run_id"]):
                    raise ValueError("Existing daily metrics have a foreign run identity")
            done_aids.add(row["aid"])
            if row["status"] == "skipped":
                if (row.get("skip_kind") != "failed_after_retries"
                        or row.get("attempts") != 6 or row.get("observed_behavior") is not False):
                    raise ValueError("Existing skipped metric has an invalid retry receipt")
                skipped_aids.add(row["aid"])
            else:
                ok_count += 1
    remaining = [aid for aid in agents if aid not in done_aids]
    print(f"[Day {day_idx} {day_str}] processing {len(remaining)} agents with {workers} workers")

    # 메트릭 jsonl append 모드
    lock = Lock()
    fail_list: list[dict] = []
    err_count = 0
    t_start = time.time()
    last_progress = 0

    # write 실패 retry + 매 500 agent마다 checkpoint snapshot
    def _safe_write(fp, line):
        try:
            fp.write(line)
            fp.flush()
            os.fsync(fp.fileno())
            return True
        except OSError:
            # Never repeat an append that may already be partially written.
            # Leave the DB outbox and original bytes for verified recovery.
            return False

    attempt_path = CHECK_DIR / f"attempts_{day_str}.jsonl"
    from collections import Counter
    from day_resume import read_metric_rows
    attempts = read_metric_rows(attempt_path)
    attempt_counts = Counter()
    last_errors = {}
    for attempt in attempts:
        if (attempt.get("aid") not in agents or attempt.get("status") not in {"error", "no_persona"}):
            raise ValueError("Existing attempt log has an invalid agent or status")
        consumed = attempt.get('attempts_consumed', 1)
        if type(consumed) is not int or not 1 <= consumed <= 6:
            raise ValueError('Existing attempt log has an invalid consumed budget')
        attempt_counts[attempt["aid"]] = min(6, attempt_counts[attempt["aid"]] + consumed)
        last_errors[attempt["aid"]] = attempt.get("error", attempt["status"])
        if attempt_counts[attempt["aid"]] > 6:
            raise ValueError("Existing attempt log exceeds the six-attempt budget")
    # Grounded study decisions need exact evidence and per-order candidates.
    # Retry only the failed agents at low concurrency; never invent a citation
    # or silently substitute a POI. This also resolves transient DB deadlocks.
    max_rounds = 6  # initial attempt plus at most five retries per agent-day
    pending = remaining
    import checkpoint_control
    checkpoint_control.save_if_due(OUT_DIR, day_str)
    for round_idx in range(max_rounds):
        if not pending:
            break
        exhausted = [aid for aid in pending if attempt_counts[aid] >= max_rounds]
        pending = [aid for aid in pending if attempt_counts[aid] < max_rounds]
        for aid in exhausted:
            skipped = record_skipped_agent_day(aid, today, attempt_counts[aid], last_errors[aid])
            with metrics_path.open("a", encoding="utf-8") as fp:
                if not _safe_write(fp, json.dumps(skipped, ensure_ascii=False) + "\n"):
                    raise RuntimeError("skipped result persistence failed; reconcile database outbox")
            done_aids.add(aid)
            skipped_aids.add(aid)
        if not pending:
            break
        fail_list = []
        round_workers = workers if round_idx == 0 else min(workers, 2)
        with ThreadPoolExecutor(max_workers=round_workers) as ex:
            # Keep only a worker-sized queue so quiescent graph backup can run.
            todo = iter(pending)
            futures = {ex.submit(process_one, aid, today, day_idx): aid
                       for aid in [next(todo, None) for _ in range(round_workers)] if aid is not None}
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                fut = next(iter(ready))
                res = fut.result()
                if res.get("aid") != futures.pop(fut):
                    raise ValueError("Agent result identity mismatch")
                with lock:
                    try:
                        path = metrics_path if res.get("status") == "ok" else attempt_path
                        with path.open("a", encoding="utf-8") as fp:
                            if not _safe_write(fp, json.dumps(res, ensure_ascii=False) + "\n"):
                                raise OSError("agent result write failed")
                    except OSError as e:
                        raise RuntimeError("agent result persistence failed; reconcile database outbox") from e
                    if res.get("status") == "ok":
                        if res["aid"] in done_aids:
                            raise ValueError("Duplicate completed agent result")
                        ok_count += 1
                        done_aids.add(res["aid"])
                    else:
                        fail_list.append(res)
                        attempt_counts[res["aid"]] = min(
                            max_rounds, attempt_counts[res["aid"]] + res.get('attempts_consumed', 1))
                        last_errors[res["aid"]] = res.get("error", res["status"])
                total_done = len(done_aids) + len(fail_list)
                if round_idx == 0 and total_done - last_progress >= max(20, len(remaining)//20):
                    elapsed = time.time() - t_start
                    rate = (total_done - (len(agents) - len(remaining))) / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - (total_done - (len(agents) - len(remaining)))) / rate if rate > 0 else 0
                    print(f"  {total_done}/{len(agents)} (ok={ok_count}, retry={len(fail_list)}) "
                          f"@ {rate:.1f}/s, ETA {eta:.0f}s")
                    last_progress = total_done
                if ok_count and ok_count % 500 == 0:
                    atomic_json(done_path, sorted(done_aids))
                if not checkpoint_control.due(OUT_DIR):
                    aid = next(todo, None)
                    if aid is not None:
                        futures[ex.submit(process_one, aid, today, day_idx)] = aid
                if not futures and checkpoint_control.due(OUT_DIR):
                    checkpoint_control.save_if_due(OUT_DIR, day_str)
                    for _ in range(round_workers):
                        aid = next(todo, None)
                        if aid is not None:
                            futures[ex.submit(process_one, aid, today, day_idx)] = aid
        pending = [res["aid"] for res in fail_list]
        if pending:
            if round_idx < max_rounds - 1:
                print(f"  [retry {round_idx+1}/5] {len(pending)} agents remain", flush=True)
    for aid in pending:
        if attempt_counts[aid] != max_rounds:
            raise RuntimeError("Agent retry accounting is inconsistent")
        skipped = record_skipped_agent_day(aid, today, attempt_counts[aid], last_errors[aid])
        with metrics_path.open("a", encoding="utf-8") as fp:
            if not _safe_write(fp, json.dumps(skipped, ensure_ascii=False) + "\n"):
                raise RuntimeError("skipped result persistence failed; reconcile database outbox")
        done_aids.add(aid)
        skipped_aids.add(aid)
    err_count = len(skipped_aids)

    try:
        atomic_json(done_path, sorted(done_aids))
        atomic_json(failed_path, fail_list)
    except OSError as e:
        print(f"  [warn] final checkpoint write failed: {e}")

    if done_aids != set(agents):
        raise RuntimeError(f"incomplete agent day {day_str}: terminal receipts are missing")

    agent_elapsed = time.time() - t_start
    print(
        f"[Day {day_idx} {day_str}] agent phase done in {agent_elapsed:.0f}s "
        f"— ok={ok_count}, skipped={err_count}"
    )
    timing_report = _write_timing_diagnostics(day_str, metrics_path)
    day_result = {
        "day": day_str,
        "ok": ok_count,
        "err": err_count,
        "skipped": err_count,
        "agent_elapsed_sec": agent_elapsed,
        "night2_elapsed_sec": 0.0,
        "elapsed_sec": agent_elapsed,
        "timing_top": (timing_report.get("bottleneck_rank") or [])[:10],
    }

    # Terminal night skips are missing interactions; they do not block the date.
    from night_completion import complete_night
    day_result.update(complete_night(today, cohort, skipped_aids, workers, OUT_DIR))

    day_result["elapsed_sec"] = time.time() - t_start
    print(
        f"[Day {day_idx} {day_str}] done in {day_result['elapsed_sec']:.0f}s "
        f"(agent={day_result['agent_elapsed_sec']:.0f}s, "
        f"Night2={day_result['night2_elapsed_sec']:.0f}s)"
    )
    return day_result
