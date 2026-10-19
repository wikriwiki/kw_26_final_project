def classify_intent(pair_key: tuple[str, str], data: dict, max_retry: int = 5) -> dict | None:
    from no_smoking_context import configured_context, clear_llm_scope
    runtime = configured_context()
    token = None
    try:
        if runtime:
            from interview_evidence import begin_evidence, set_evidence_stage, archive_interaction
            from evidence_integrity import digest
            from experience_provenance import source_fingerprint
            out = Path(os.environ.get('SIM_OUTPUT_DIR', os.path.expanduser('~/sim_output')))
            token = begin_evidence(
                out, os.environ.get('SIM_RUN_ID') or str(out.resolve()), runtime.arm,
                str(data['simulation_day']), list(pair_key), digest(runtime.agent_ids),
                source_fingerprint(), context=json.loads(json.dumps(data, default=str)),
            )
            set_evidence_stage('night_intent')
        result = _classify_intent(pair_key, data, max_retry)
        if token is not None and 'error' not in result:
            result['interview_evidence'] = archive_interaction(result)
        return result
    finally:
        clear_llm_scope()
        if token is not None:
            from interview_evidence import clear_evidence
            clear_evidence(token)

def _classify_intent(pair_key: tuple[str, str], data: dict, max_retry: int = 5) -> dict | None:
    """한 쌍에 대해 의도 분류 LLM 호출. data에 보관된 score(exp/rel/urg)를
    결과에 함께 실어 importance 계산에 사용."""
    from no_smoking_context import begin_llm_scope, configured_context
    begin_llm_scope("|".join(pair_key), data.get("simulation_day"), "night_intent")
    grounded_experiment = configured_context() is not None
    user = build_user_block(pair_key, data)
    if grounded_experiment:
        user = _number_evidence_lines(
            user, exclude_prefixes=('위 입력만 근거로', '- role:', '- agent_id:'),
        )
    evidence_lines = _evidence_lines(user) if grounded_experiment else {}
    if grounded_experiment and not evidence_lines:
        raise ValueError('Night grounded prompt has no factual evidence lines')
    system_prompt = SYSTEM_NIGHT if grounded_experiment else SYSTEM_PROMPT
    from grounded_schema import night_format
    from execution_errors import fatal_dispatch_error
    schema = night_format(evidence_lines, pair_key) if grounded_experiment else None
    last_err = None
    raw = ''
    from night_recovery import ATTEMPT_OFFSET
    from grounded_schema import rejected_response_feedback
    for attempt in range(max_retry + 1):
        temp = (0.2 if attempt == 0 else 0.1) if grounded_experiment else 0.3 + 0.2 * attempt
        try:
            resp = _llm_call(
                None, system_prompt, user + (
                    f'\n[야간 분류 시도 {ATTEMPT_OFFSET.get() + attempt + 1}/6]\n'
                    + rejected_response_feedback(raw, last_err,
                        'evidence_ref에는 입력에 있는 사실 줄 번호 한 개만 쓰세요. '
                        '쉼표로 여러 번호를 합치지 마세요. 참가자 두 명을 바꾸지 마세요.')
                    if grounded_experiment and last_err else ''),
                temperature=temp, max_tokens=900 if grounded_experiment else 600,
                **({'response_format': schema} if grounded_experiment else {}),
            )
            raw = resp.choices[0].message.content
            data_json = json.loads(_extract_first_json(raw) if grounded_experiment else _extract_json(raw))
            if grounded_experiment:
                ref = data_json.get('evidence_ref')
                if not isinstance(ref, str) or ref not in evidence_lines:
                    raise ValueError(f'evidence_ref={ref!r}: exactly one numbered factual input line is required')
                data_json['evidence_quote'] = evidence_lines[ref]
                validate_stated_reason(data_json, user)
            parsed = IntentOutput.model_validate(data_json)
            if (parsed.initiator_id, parsed.recipient_id) != pair_key:
                raise ValueError('Night response changed the matched participants')
            if parsed.intent != '약속' and parsed.plan_signal.should_inject:
                raise ValueError('Only an appointment can inject a future plan')
            return {
                "intent": parsed.intent,
                "initiator_id": parsed.initiator_id,
                "recipient_id": parsed.recipient_id,
                "topic_type": parsed.topic_type,
                "topic_value": parsed.topic_value,
                "should_inject": parsed.plan_signal.should_inject,
                "target_day_offset": parsed.plan_signal.target_day_offset,
                "target_time": parsed.plan_signal.target_time,
                "meeting_location_hint": parsed.plan_signal.meeting_location_hint,
                "reasoning": parsed.reasoning,   # ← Conversation.reasoning + Memory.summary 로 흐름
                "evidence_ref": parsed.evidence_ref,
                "evidence_quote": parsed.evidence_quote,
                "statement_kind": "model_inferred_interaction",
                # 매칭 점수(importance 계산용 — 노션 §9)
                "interaction_score": data.get("score", 0.0),
                "exposure_score": data.get("exp", 0.0),
                "relationship_score": data.get("rel", 0.0),
                "urgency_score": data.get("urg", 0.0),
                "threshold_used": data.get("threshold_used"),
                "ambient_threshold_applied": bool(data.get("ambient_threshold_applied")),
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "attempt": attempt,
            }
        except Exception as e:
            if fatal_dispatch_error(e):
                raise
            last_err = e
    return {"error": str(last_err)[:200], "pair": list(pair_key)}

def write_conversations(day: date, results: list[dict], skipped=None):
    """의도 분류 결과 → Conversation 노드 + intent별 후속 엣지·노드 (노션 §4·§5·§9·§10).

    intent별 처리:
      - 약속: Conversation(+ plan_signal) → meeting_hint가 POI이면 MENTIONS_POI
      - 이슈: Conversation → Memory{rumor} + REMEMBERS + FROM_CONVERSATION
              + topic_type=policy면 ABOUT_POLICY 엣지
      - 추천: Conversation → Memory{rumor} + REMEMBERS + FROM_CONVERSATION
              + MENTIONS_POI + KNOWS_POI{rumor} MERGE + Memory.ABOUT_POI
      - 기타: Conversation만 (노션 §4 — 기타는 Conversation만 적재)
    """
    base_rows, rumor_rows, recommend_rows, issue_rows, appt_rows = [], [], [], [], []

    # recipient별 Memory id 카운터 (MEM_RUMOR_<recipient>_D<day>_<n>)
    mem_seq: dict[str, int] = {}
    day_iso = day.isoformat()
    day_tag = day_iso.replace("-", "")   # D20260515 식

    for r in results:
        if "error" in r:
            continue
        intent = r["intent"]
        cid = f"conv_{uuid.uuid4().hex[:16]}"
        r['conversation_id'] = cid

        # Conversation 베이스 (노션 §4 — 모든 intent 공통)
        base_rows.append({
            "cid": cid, "day": day_iso,
            "intent": intent,
            "initiator": r["initiator_id"],
            "recipient": r["recipient_id"],
            "topic_type": r["topic_type"],
            "topic_value": r.get("topic_value"),
            "should_inject": bool(r.get("should_inject")),
            "target_day_offset": r.get("target_day_offset"),
            "target_time": r.get("target_time"),
            "meeting_location_hint": r.get("meeting_location_hint"),
            # 사고과정 흔적 (인터뷰 인용용)
            "reasoning": r.get("reasoning"),
            "evidence_quote": r.get("evidence_quote"),
            "statement_kind": r.get("statement_kind", "model_inferred_interaction"),
            # Night pair selection debug fields
            "interaction_score": r.get("interaction_score"),
            "exposure_score": r.get("exposure_score"),
            "relationship_score": r.get("relationship_score"),
            "urgency_score": r.get("urgency_score"),
            "threshold_used": r.get("threshold_used"),
            "ambient_threshold_applied": bool(r.get("ambient_threshold_applied")),
        })

        # intent별 분기
        if intent in ("이슈", "추천"):
            # Memory{rumor} 생성 — 두 intent 모두 recipient에 적재 (노션 §5)
            recipient = r["recipient_id"]
            mem_seq[recipient] = mem_seq.get(recipient, 0) + 1
            mem_id = f"MEM_RUMOR_{recipient}_D{day_tag}_{mem_seq[recipient]}"

            # importance = urgency × 0.6 + relationship × 0.4 (노션 §9)
            imp = (r.get("urgency_score", 0.0) * 0.6
                   + r.get("relationship_score", 0.0) * 0.4)
            imp = round(imp, 2)

            rumor_rows.append({
                "cid": cid, "mem_id": mem_id,
                "initiator": r["initiator_id"],
                "recipient": recipient,
                "importance": imp,
            })

            if intent == "추천":
                recommend_rows.append({
                    "cid": cid, "mem_id": mem_id,
                    "recipient": recipient,
                    "topic_type": r["topic_type"],
                    "topic_value": r.get("topic_value"),
                })
            else:  # 이슈
                issue_rows.append({
                    "cid": cid,
                    "topic_type": r["topic_type"],
                    "topic_value": r.get("topic_value"),
                })
        elif intent == "약속":
            appt_rows.append({
                "cid": cid,
                "meeting_hint": r.get("meeting_location_hint"),
            })
        # "기타"는 base만 — 노션 §4

    by_intent: dict[str, int] = {}
    for row in base_rows:
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    stats = {
        "created": len(base_rows),
        "by_intent": by_intent,
        "rumor_memory": len(rumor_rows),
        "recommend_extra": len(recommend_rows),
        "issue_extra": len(issue_rows),
        "appointment_extra": len(appt_rows),
    }
    from no_smoking_context import configured_context
    runtime = configured_context()
    with driver_session() as session:
        with session.begin_transaction() as tx:
            for query, rows in ((CREATE_CONVERSATION_CYPHER, base_rows),
                                (LINK_RUMOR_MEMORY_CYPHER, rumor_rows),
                                (LINK_RECOMMEND_EXTRA_CYPHER, recommend_rows),
                                (LINK_ISSUE_EXTRA_CYPHER, issue_rows),
                                (LINK_APPOINTMENT_EXTRA_CYPHER, appt_rows)):
                if rows:
                    tx.run(query, rows=rows).consume()
            if runtime:
                from interview_evidence import commit_night_evidence
                import night_store
                out = Path(os.environ.get('SIM_OUTPUT_DIR', os.path.expanduser('~/sim_output')))
                reference = commit_night_evidence(out, os.environ.get('SIM_RUN_ID') or str(out.resolve()),
                                                 runtime.arm, day.isoformat(), results, skipped=skipped)
                night_store.save(tx, day, runtime, {'processed': len(results), 'errors': 0,
                    'skipped': len(skipped or []), 'matched': len(results) + len(skipped or []),
                    'write': stats, 'evidence_ref': reference})
                stats['evidence_ref'] = reference
            tx.commit()
    return stats

def run_intent_classification(
    day: date,
    pairs: list[dict],
    workers: int = 16,
    verbose: bool = True,
) -> dict:
    import os
    # 기본을 엄격으로 둔다. 이미 적재된 Night2 위에 덧쓰는 것은 침묵하는 손상이고,
    # 멀쩡한 날을 건너뛰는 것도 마찬가지다. 되살리기(resume)는 사람이 그 상황을
    # 확인하고 SIM_STRICT_COMPLETION=0 으로 명시해야 열린다.
    strict = os.environ.get("SIM_STRICT_COMPLETION", "1") != "0"
    if not pairs:
        return {"processed": 0}
    # 멱등성: 같은 day Conversation이 이미 90% 이상 적재됐으면 skip
    # (resume / 모델 swap 후 재실행 시 Night2 중복 방지)
    try:
        with driver_session() as s:
            existing = s.run(
                "MATCH (c:Conversation) WHERE c.day = date($d) RETURN count(c) AS n",
                d=day.isoformat()
            ).single()["n"]
        if strict and existing:
            raise RuntimeError("Strict Night2 requires no existing conversations")
        if not strict and existing >= int(0.9 * len(pairs)):
            if verbose:
                print(f"[Intent] day {day}: {existing}/{len(pairs)} 이미 적재됨 — Night2 skip")
            return {"processed": 0, "skipped": True, "existing": existing,
                    "write": {"created": 0, "by_intent": {}}}
    except Exception as e:
        if strict:
            raise
        if verbose:
            print(f"[Intent] idempotency 체크 실패 (계속 진행): {e}")
    t0 = time.time()
    if verbose:
        print(f"[Intent] fetching pair data for {len(pairs)} pairs ...")
    pair_data = fetch_pair_data(pairs, day)
    if len(pair_data) != len(pairs):
        raise RuntimeError('Night pair input is missing or duplicated; cannot account for every pair')
    if verbose:
        print(f"  pair data fetched: {len(pair_data)}")

    if verbose:
        print(f"[Intent] LLM classification with {workers} workers ...")
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for (a, b), d in pair_data.items():
            from night_recovery import classify_recoverable
            futs[ex.submit(classify_recoverable, classify_intent, (a, b), d)] = (a, b)
        for fut in as_completed(futs):
            results.append(fut.result())
            completed += 1
            if verbose and completed % max(10, len(pair_data)//10) == 0:
                print(f"  {completed}/{len(pair_data)} ({time.time()-t0:.0f}s)")

    ok = [r for r in results if "error" not in r and r.get('status') != 'skipped']
    skipped = [r for r in results if r.get('status') == 'skipped']
    err = [r for r in results if r not in ok and r not in skipped]
    if err or len(ok) + len(skipped) != len(pairs):
        raise RuntimeError(f'Unaccounted Night pair results: {len(err)}')
    if verbose:
        print(f"[Intent] LLM done: {len(ok)} ok, {len(err)} err ({time.time()-t0:.0f}s)")

    write_stats = write_conversations(day, ok, skipped=skipped)
    evidence_ref = None
    from no_smoking_context import configured_context
    runtime = configured_context()
    if runtime:
        evidence_ref = write_stats['evidence_ref']
    if verbose:
        print(f"[Intent] adapted: {write_stats}")
    return {"processed": len(ok), "errors": 0, 'skipped': len(skipped), 'matched': len(pairs),
            "write": write_stats, "elapsed": time.time()-t0,
            "samples": ok[:5], "evidence_ref": evidence_ref}
