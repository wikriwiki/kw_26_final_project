def commit_night_evidence(run_dir, run_id, arm, day, results, skipped=None):
    """Called only after successful DB writes; never mutates a graph itself."""
    root = Path(run_dir).resolve()
    day = iso_day(str(day))
    clean = json_value(results)
    if not isinstance(clean, list):
        raise EvidenceError("night results must be a list")
    conversation_ids = set()
    for result in clean:
        reference = result.get("interview_evidence")
        if not reference:
            raise EvidenceError("night result lacks interaction evidence")
        archived, _ = _load_journal(root, reference, expected_kind="interaction")
        if archived["run_id"] != run_id or archived["arm"] != arm or archived["day"] != day:
            raise EvidenceError("night result belongs to another run or day")
        cid = result.get("conversation_id")
        # A classifier may return no social action, so no Conversation is
        # necessarily written. Any written conversation must have a unique ID.
        if cid is not None:
            if not isinstance(cid, str) or not cid or cid in conversation_ids:
                raise EvidenceError("duplicate or invalid conversation identity")
            conversation_ids.add(cid)
        original = {key: value for key, value in result.items() if key not in {"interview_evidence", "conversation_id"}}
        if original != archived["interaction"]:
            raise EvidenceError("night result changed after classification archive")
        result["evidence_agent_ids"] = archived["agent_ids"]
    from night_recovery import verify_skipped
    missing = [verify_skipped(root, value) for value in (skipped or [])]
    if any((v['run_id'], v['arm'], v['day']) != (run_id, arm, day) for v in missing):
        raise EvidenceError('Foreign skipped night interaction')
    success_pairs = {tuple(sorted(v['evidence_agent_ids'])) for v in clean}
    skip_pairs = {tuple(sorted(v['pair'])) for v in missing}
    if len(skip_pairs) != len(missing) or success_pairs & skip_pairs:
        raise EvidenceError('Duplicate terminal night pair')
    record = seal({"schema_version": 1, "kind": "completed_night_evidence",
                   "record_id": "EV_" + uuid4().hex, "recorded_at": now(),
                   "run_id": run_id, "arm": arm, "day": day, "results": clean,
                   "skipped": missing})
    relative = f"evidence/night/{day}_{record['record_id']}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(canonical(record))
        output.flush()
        os.fsync(output.fileno())
    return {"schema_version": 1, "path": relative, "record_id": record["record_id"],
            "integrity_sha256": record["integrity_sha256"]}

def _night_items(root, agent_id, day, identity, night_reader=None):
    loaded = night_reader(day, agent_id) if night_reader else _read_night_index(root, day)
    if loaded is None:
        return [], False
    marker, index = loaded
    run_id, arm, cohort_sha, source_sha = identity
    if (marker.get("run_id") != run_id or marker.get("arm") != arm
            or marker.get("day") != day or marker.get("status") != "complete"):
        raise EvidenceError("night completion marker identity/status mismatch")
    items = []
    for result in index["results"]:
        if agent_id not in result.get("evidence_agent_ids", []):
            continue
        archived, records = _load_journal(root, result["interview_evidence"], expected_kind="interaction")
        if (archived["run_id"], archived["arm"], archived["cohort_sha256"], archived["source_sha256"]) != identity or archived["day"] != day:
            raise EvidenceError("foreign interaction in night index")
        if sorted(result["evidence_agent_ids"]) != archived["agent_ids"] or agent_id not in archived["agent_ids"]:
            raise EvidenceError("night participant index differs from journal")
        value = {"conversation_id": result.get("conversation_id"),
                 "participants": archived["agent_ids"], "interaction": archived["interaction"],
                 "reason_status": "public_model_statement_not_verified_cause",
                 "experience_status": "recorded_social_interaction" if result.get("conversation_id") else "no_recorded_conversation"}
        value['quote_validation'] = _quote_validation(archived['interaction'], records)
        items.append({"evidence_id": archived["record_id"], "agent_id": agent_id, "day": day,
                      "kind": "social_interaction", "text": canonical(value), "value": value,
                      "source_ref": result["interview_evidence"]})
    from night_recovery import verify_skipped
    for skipped in index.get('skipped', []):
        if agent_id not in skipped['pair']:
            continue
        value = verify_skipped(root, skipped)
        if (value['run_id'], value['arm'], value['cohort_sha256'], value['source_sha256']) != identity or value['day'] != day:
            raise EvidenceError('Foreign skipped night receipt')
        items.append({'evidence_id': 'NS_' + value['integrity_sha256'],
                      'agent_id': agent_id, 'day': day, 'kind': 'skipped_social_interaction',
                      'text': canonical(value), 'value': value,
                      'source_ref': {'journal_refs': value['journals']}})
    return items, True
