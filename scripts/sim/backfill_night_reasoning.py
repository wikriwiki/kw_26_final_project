"""특정 Day의 Night Phase 2 Conversation에 reasoning 필드만 사후 LLM으로 채움.

목적: 1차 풀런 때 night_intent_llm.classify_intent 가 reasoning 키를 dict에
누락시켰던 버그(이후 fix됨)로, 그 시점에 적재된 Conversation 노드의
reasoning 컬럼이 null인 케이스를 사후 보강.

원칙:
- **메타데이터(intent, initiator_id, recipient_id, topic_type, topic_value,
  plan_signal, day) 절대 변경 X** — Day N+ 시뮬이 이미 이 데이터를 봤기 때문.
- reasoning 컬럼만 SET. id·엣지 모두 보존.

LLM 입력: 두 페르소나 + 매칭 점수 + 그 시점 매칭된 intent/topic.
LLM 출력: reasoning (1~2문장).

CLI:
  python scripts/sim/backfill_night_reasoning.py --day 2026-05-01 --workers 24
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402


SYSTEM_PROMPT = """너는 사회적 상호작용 분류 결과를 보고 그 이유를 1~2문장으로 설명하는 분석가다.
입력으로 두 agent의 페르소나 + 매칭 점수 + 이미 분류된 intent/topic 가 주어진다.
**왜 이 intent로 분류됐는지 페르소나·일정·매칭점수 근거로 1~2문장 reasoning만 출력**한다.
JSON·코드펜스 금지. 자연어 1~2문장만.

분류된 intent별 reasoning 방향:
- 약속: 누가 누구에게 어떤 사유로 약속을 잡았는지 (친밀도·일정 겹침 등)
- 이슈: 어떤 정책·뉴스가 화제가 되었는지 (정책 노출 등)
- 추천: 누가 어떤 장소·카테고리를 추천한 사유 (단골 만족도 등)
- 기타: 일상 잡담의 이유 (페르소나·일정 무관함 등)
/no_think"""


def build_prompt(row: dict) -> str:
    """Conversation 메타 + 두 agent 페르소나로 user_block 구성."""
    a, b = row["a"], row["b"]
    a_life = (a.get("life") or "")[:80]
    b_life = (b.get("life") or "")[:80]
    return f"""### MATCHING_SCORE
exposure={row.get('exp', 0):.2f} relationship={row.get('rel', 0):.2f} urgency={row.get('urg', 0):.2f}

### AGENT_A (initiator)
- id: {row['initiator']}
- job: {a.get('job') or '-'}
- lifestyle: {a_life}

### AGENT_B (recipient)
- id: {row['recipient']}
- job: {b.get('job') or '-'}
- lifestyle: {b_life}

### CLASSIFIED
- intent: {row['intent']}
- topic_type: {row.get('topic_type')} / topic_value: {row.get('topic_value') or '-'}

위 입력 기준 reasoning 1~2문장만. JSON 금지. /no_think"""


def fetch_targets(day: str) -> list[dict]:
    """reasoning이 null인 Day의 Conversation + 두 agent 페르소나 fetch."""
    with driver_session() as s:
        rows = s.run("""
            MATCH (c:Conversation {day: date($d)})
            WHERE c.reasoning IS NULL
            MATCH (a:Agent {id: c.initiator_id}), (b:Agent {id: c.recipient_id})
            OPTIONAL MATCH (a)-[:KNOWS]-(b)
            RETURN c.id AS cid, c.intent AS intent,
                   c.initiator_id AS initiator, c.recipient_id AS recipient,
                   c.topic_type AS topic_type, c.topic_value AS topic_value,
                   a.personal_job_raw AS a_job,
                   a.personality_lifestyle_raw AS a_life,
                   b.personal_job_raw AS b_job,
                   b.personality_lifestyle_raw AS b_life,
                   0.0 AS exp, 0.0 AS rel, 0.0 AS urg
        """, d=day).data()
    out = []
    for r in rows:
        out.append({
            "cid": r["cid"],
            "intent": r["intent"],
            "initiator": r["initiator"],
            "recipient": r["recipient"],
            "topic_type": r["topic_type"],
            "topic_value": r["topic_value"],
            "a": {"job": r["a_job"], "life": r["a_life"]},
            "b": {"job": r["b_job"], "life": r["b_life"]},
            "exp": r["exp"], "rel": r["rel"], "urg": r["urg"],
        })
    return out


def _clean(text: str) -> str:
    # <think> 제거, 줄바꿈/특수 정리, 첫 200자
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = t.strip().replace("\n", " ")
    return t[:300]


def call_llm(row: dict, max_retry: int = 1) -> str | None:
    user = build_prompt(row)
    for attempt in range(max_retry + 1):
        try:
            resp = _llm_call(None, SYSTEM_PROMPT, user,
                             temperature=0.6, max_tokens=200)
            return _clean(resp.choices[0].message.content)
        except Exception:
            if attempt == max_retry:
                return None
    return None


def update_batch(updates: list[dict]):
    if not updates:
        return
    with driver_session() as s:
        s.run("""
            UNWIND $rows AS r
            MATCH (c:Conversation {id: r.cid})
            SET c.reasoning = r.reasoning
        """, rows=updates).consume()
        # rumor Memory.summary도 같이 update (이미 적재된 케이스)
        s.run("""
            UNWIND $rows AS r
            MATCH (c:Conversation {id: r.cid})
            OPTIONAL MATCH (m:Memory {type:'rumor'})-[:FROM_CONVERSATION]->(c)
            WHERE m IS NOT NULL AND m.summary IS NULL
            SET m.summary = r.reasoning
        """, rows=updates).consume()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-05-01")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    print(f"[1/3] reasoning=null 인 Conversation fetch (day={args.day}) ...",
          file=sys.stderr)
    targets = fetch_targets(args.day)
    print(f"  → {len(targets):,}건 대상", file=sys.stderr)
    if not targets:
        print("  → 대상 없음. 종료.", file=sys.stderr)
        return

    print(f"[2/3] LLM 호출 ({args.workers} worker) ...", file=sys.stderr)
    t0 = time.time()
    updates: list[dict] = []
    completed = 0
    batch_n = max(1, len(targets) // 20)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_llm, row): row for row in targets}
        for fut in as_completed(futs):
            row = futs[fut]
            r = fut.result()
            if r:
                updates.append({"cid": row["cid"], "reasoning": r})
            completed += 1
            if completed % batch_n == 0:
                elapsed = time.time() - t0
                eta = elapsed / completed * (len(targets) - completed)
                print(f"  {completed:,}/{len(targets):,} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)", file=sys.stderr)
            # batch flush
            if len(updates) >= args.batch:
                update_batch(updates)
                updates = []

    if updates:
        update_batch(updates)

    print(f"[3/3] DONE — {completed:,}건 처리, {time.time()-t0:.0f}s",
          file=sys.stderr)
    # 검증
    with driver_session() as s:
        r = s.run("""
            MATCH (c:Conversation {day: date($d)})
            RETURN count(*) AS total,
                   sum(CASE WHEN c.reasoning IS NOT NULL THEN 1 ELSE 0 END) AS has_r
        """, d=args.day).single()
        pct = r["has_r"] / max(r["total"], 1) * 100
        print(f"  최종 reasoning 적재율: {r['has_r']:,}/{r['total']:,} ({pct:.1f}%)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
