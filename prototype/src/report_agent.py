"""
주간 리포트 에이전트 — ReportAgent

매 주(1라운드) 종료 시 구조화된 분석 보고서를 생성.
Seoul_BigData_PLAN.md §5단계: 보고서 생성 참조.

출력:
  - output/reports/week_NN.json  (구조화 데이터)
  - output/reports/week_NN.md    (사람이 읽는 텍스트 보고서)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

from config import OUTPUT_DIR


REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_weekly_report(
    week: int,
    agents: list[dict],
    week_logs: dict,          # {agent_id: [[day0_actions], [day1_actions], ...]}
    env_summary: dict,        # from EnvironmentManager.get_week_summary()
    prev_report: dict = None, # 전주 리포트 (비교용)
    graph_mgr = None,         # GraphMemoryManager (커뮤니티 분석용)
) -> dict:
    """주간 분석 보고서 생성.

    Returns: 리포트 dict (JSON 저장 + 다음주 비교용)
    """
    # ── 1. 기본 통계 ──
    total_spending = 0
    total_actions = 0
    segment_stats = defaultdict(lambda: {
        "count": 0,
        "spending": 0,
        "actions": 0,
        "action_types": Counter(),
        "industries": Counter(),
        "visited_dongs": set(),
        "satisfactions": [],
    })

    agent_summaries = {}  # {agent_id: 개별 요약}

    for agent in agents:
        aid = agent["agent_id"]
        seg = agent["segment"]
        days = week_logs.get(aid, [])

        agent_spending = 0
        agent_actions = 0
        agent_industries = Counter()
        agent_action_types = Counter()
        agent_dongs = set()
        agent_sats = []
        agent_triggers = Counter()  # 이 에이전트에 영향을 준 환경 이벤트 카운트
        agent_new_industries = set()  # 이번 주 처음 시도한 업종

        for day_actions in days:
            for act in day_actions:
                amount = act.get("amount", 0)
                agent_spending += amount
                total_spending += amount

                if act["type"] == "외출_소비":
                    agent_actions += 1
                    total_actions += 1
                    if act.get("industry"):
                        agent_industries[act["industry"]] += 1
                    if act.get("dong"):
                        agent_dongs.add(act["dong"])
                    if act.get("satisfaction"):
                        agent_sats.append(act["satisfaction"])

                    # triggered_by 수집 — 환경 에이전트 영향 추적
                    for trigger in act.get("triggered_by", []):
                        agent_triggers[trigger] += 1
                    # 뉴스 탐색으로 새 업종 시도 감지
                    if any("뉴스탐색" in t for t in act.get("triggered_by", [])):
                        agent_new_industries.add(act.get("industry", ""))

                # 행동 유형 분류
                action_type = _classify_action(act, agent)
                agent_action_types[action_type] += 1

        seg_st = segment_stats[seg]
        seg_st["count"] += 1
        seg_st["spending"] += agent_spending
        seg_st["actions"] += agent_actions
        seg_st["action_types"] += agent_action_types
        seg_st["industries"] += agent_industries
        seg_st["visited_dongs"].update(agent_dongs)
        seg_st["satisfactions"].extend(agent_sats)

        agent_summaries[aid] = {
            "segment": seg,
            "spending": agent_spending,
            "actions": agent_actions,
            "top_industry": agent_industries.most_common(1)[0][0] if agent_industries else None,
            "all_industries": list(agent_industries.keys()),
            "satisfaction": round(np.mean(agent_sats), 2) if agent_sats else 0.5,
            "dongs_visited": len(agent_dongs),
            "action_types": dict(agent_action_types),
            "triggers": dict(agent_triggers),
            "new_explores": list(agent_new_industries),
        }

    # ── 2. 세그먼트별 분석 ──
    segment_reports = {}
    for seg, st in segment_stats.items():
        n = max(st["count"], 1)
        all_sats = st["satisfactions"]
        total_type_count = sum(st["action_types"].values()) or 1

        segment_reports[seg] = {
            "agent_count": st["count"],
            "total_spending": st["spending"],
            "avg_spending": round(st["spending"] / n),
            "avg_actions_per_agent": round(st["actions"] / n, 1),
            "avg_satisfaction": round(np.mean(all_sats), 2) if all_sats else 0.5,
            "top_industries": dict(st["industries"].most_common(5)),
            "behavior_distribution": {
                k: round(v / total_type_count * 100, 1)
                for k, v in st["action_types"].most_common()
            },
            "unique_dongs": len(st["visited_dongs"]),
        }

    # ── 3. 전주 대비 변화 ──
    deltas = {}
    behavior_changes = {}  # 전후 행동 패턴 비교
    if prev_report:
        prev_segs = prev_report.get("segment_reports", {})
        for seg in segment_reports:
            prev = prev_segs.get(seg, {})
            cur = segment_reports[seg]
            deltas[seg] = {
                "spending_delta": cur["avg_spending"] - prev.get("avg_spending", 0),
                "satisfaction_delta": round(
                    cur["avg_satisfaction"] - prev.get("avg_satisfaction", 0.5), 2
                ),
                "actions_delta": round(
                    cur["avg_actions_per_agent"] - prev.get("avg_actions_per_agent", 0), 1
                ),
            }

            # ── 전후 행동 패턴 비교 ──
            prev_top = list(prev.get("top_industries", {}).keys())[:5]
            cur_top = list(cur["top_industries"].keys())[:5]
            new_industries = [i for i in cur_top if i not in prev_top]
            dropped_industries = [i for i in prev_top if i not in cur_top]

            prev_behavior = prev.get("behavior_distribution", {})
            cur_behavior = cur.get("behavior_distribution", {})
            behavior_shifts = {}
            for btype in set(list(prev_behavior.keys()) + list(cur_behavior.keys())):
                prev_pct = prev_behavior.get(btype, 0)
                cur_pct = cur_behavior.get(btype, 0)
                if abs(cur_pct - prev_pct) >= 1.0:
                    behavior_shifts[btype] = round(cur_pct - prev_pct, 1)

            behavior_changes[seg] = {
                "new_top_industries": new_industries,
                "dropped_top_industries": dropped_industries,
                "behavior_shifts": behavior_shifts,
                "spending_trend": "증가" if deltas[seg]["spending_delta"] > 5000 else
                                  "감소" if deltas[seg]["spending_delta"] < -5000 else "유지",
                "satisfaction_trend": "상승" if deltas[seg]["satisfaction_delta"] > 0.03 else
                                      "하락" if deltas[seg]["satisfaction_delta"] < -0.03 else "유지",
            }

    # ── 4. 입소문 확산 ──
    recommendation_stats = _analyze_recommendations(agents, week_logs)

    # ── 5. 에이전트 인터뷰 (환경 변화에 가장 많이 영향받은 에이전트) ──
    interviews = _generate_interviews(agents, agent_summaries, week_logs, env_summary, prev_report)

    # ── 6. 커뮤니티별 소비 분석 (GraphRAG + LLM 해석) ──
    community_analysis = {}
    if graph_mgr and graph_mgr.social._communities:
        comm_data = defaultdict(lambda: {
            "members": 0, "spending": 0, "satisfactions": [],
            "industries": Counter(), "segments": Counter(),
        })
        for agent in agents:
            aid = agent["agent_id"]
            cid = graph_mgr.social.get_community(aid)
            if cid is None:
                continue
            summ = agent_summaries.get(aid, {})
            cd = comm_data[cid]
            cd["members"] += 1
            cd["spending"] += summ.get("spending", 0)
            cd["segments"][agent["segment"]] += 1
            if summ.get("satisfaction"):
                cd["satisfactions"].append(summ["satisfaction"])
            for ind in summ.get("all_industries", []):
                cd["industries"][ind] += 1

        # 상위 5개 커뮤니티
        SEG_KR = {
            "commuter": "직장인", "weekend_visitor": "주말방문객",
            "resident": "거주민", "evening_visitor": "저녁방문자",
        }
        sorted_comms = sorted(comm_data.items(),
                              key=lambda x: x[1]["members"], reverse=True)[:5]

        for cid, cd in sorted_comms:
            n = max(cd["members"], 1)
            sats = cd["satisfactions"]
            dom_seg = cd["segments"].most_common(1)[0][0] if cd["segments"] else "?"
            top_inds = dict(cd["industries"].most_common(3))

            community_analysis[cid] = {
                "members": cd["members"],
                "avg_spending": round(cd["spending"] / n),
                "avg_satisfaction": round(np.mean(sats), 2) if sats else 0.5,
                "top_industries": top_inds,
                "dominant_segment": dom_seg,
                "interpretation": "",  # LLM 해석
            }

        # LLM 해석 (일괄 호출)
        community_analysis = _interpret_communities_llm(community_analysis, SEG_KR)

    report = {
        "week": week,
        "total_spending": total_spending,
        "total_actions": total_actions,
        "agent_count": len(agents),
        "avg_spending_per_agent": round(total_spending / max(len(agents), 1)),
        "segment_reports": segment_reports,
        "deltas": deltas,
        "behavior_changes": behavior_changes,
        "env_summary": env_summary,
        "interviews": interviews,
        "recommendations": recommendation_stats,
        "community_analysis": community_analysis,
        "_agent_summaries": agent_summaries,
    }

    # ── 7. 저장 ──
    _save_json(report, week)
    _save_markdown(report, week)

    return report


def _interpret_communities_llm(community_analysis, seg_kr):
    """LLM으로 각 커뮤니티의 소비 패턴을 해석.

    LLM 불가 시 빈 interpretation을 그대로 반환.
    """
    if not community_analysis:
        return community_analysis

    import requests, json

    try:
        from llm_client import OLLAMA_URL, MODEL_NAME
        r = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=3)
        if r.status_code != 200:
            return community_analysis
    except Exception:
        return community_analysis

    # 컨텍스트 구성
    lines = []
    for cid, ca in community_analysis.items():
        seg_name = seg_kr.get(ca["dominant_segment"], ca["dominant_segment"])
        top_inds = ', '.join(list(ca["top_industries"].keys())[:3])
        lines.append(
            f"커뮤니티 {cid}: {ca['members']}명, 주로 {seg_name}, "
            f"평균소비 {ca['avg_spending']:,}원, 만족도 {ca['avg_satisfaction']}, "
            f"상위업종 {top_inds}"
        )

    ctx = "\n".join(lines)

    prompt = f"""당신은 서울 소비 트렌드 분석가입니다. 
소셜 네트워크에서 발견된 소비자 커뮤니티별 데이터를 분석하세요.

[커뮤니티 데이터]
{ctx}

[규칙]
- 각 커뮤니티에 대해 1~2문장으로 해석하세요.
- "이 그룹은 왜 이런 패턴인지", "어떤 특성의 소비자인지" 설명하세요.
- JSON 객체로 반환: {{"커뮤니티ID": "해석 문장", ...}}
- /no_think"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 500},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]

        interpretations = json.loads(raw)
        if isinstance(interpretations, dict):
            for key, text in interpretations.items():
                # 키가 문자열일 수 있으니 정수/문자열 모두 매칭
                for cid in community_analysis:
                    if str(cid) == str(key):
                        community_analysis[cid]["interpretation"] = str(text)

    except Exception:
        pass

    return community_analysis

def _classify_action(act, agent):
    """행동을 5가지 유형으로 분류."""
    if act["type"] == "재택":
        return "재택"
    dong = act.get("dong", "")
    home = agent.get("home_adm_cd", "")
    work = str(agent.get("adm_cd", ""))[:8]

    if dong == work:
        return "유지"
    elif dong == home:
        return "근거리"
    elif dong and dong != work and dong != home:
        return "전환"
    return "유지"


def _analyze_recommendations(agents, week_logs):
    """추천 전파 분석."""
    rec_given = 0
    rec_received = 0
    rec_industries = Counter()

    for agent in agents:
        aid = agent["agent_id"]
        days = week_logs.get(aid, [])
        for day_actions in days:
            for act in day_actions:
                if act.get("activity") == "recommend":
                    rec_given += 1
                    if act.get("industry"):
                        rec_industries[act["industry"]] += 1

    return {
        "total_recommendations": rec_given,
        "top_recommended_industries": dict(rec_industries.most_common(3)),
    }


def _generate_interviews(agents, agent_summaries, week_logs, env_summary, prev_report):
    """에이전트 페르소나 기반 인터뷰 — LLM으로 개인화된 감상 생성."""
    import requests

    interviews = []

    # ── 선택: 세그먼트별 상위 3명 (다양성 점수 기준) ──
    by_segment = defaultdict(list)
    for aid, summ in agent_summaries.items():
        variety_score = (
            len(summ.get("all_industries", [])) * 2
            + len(summ.get("new_explores", [])) * 5
            + summ.get("dongs_visited", 0) * 3
            + sum(summ.get("triggers", {}).values())
        )
        if prev_report:
            prev_summ = prev_report.get("_agent_summaries", {}).get(aid, {})
            prev_spend = prev_summ.get("spending", summ["spending"])
            if prev_spend > 0:
                variety_score += abs(summ["spending"] - prev_spend) / 5000
        by_segment[summ["segment"]].append((aid, summ, variety_score))

    selected = []
    for seg, candidates in by_segment.items():
        candidates.sort(key=lambda x: x[2], reverse=True)
        for aid, summ, _ in candidates[:3]:
            selected.append((aid, summ))

    SEG_NAMES = {
        "commuter": "직장인",
        "weekend_visitor": "주말방문객",
        "resident": "거주민",
        "evening_visitor": "저녁방문자",
    }

    SEG_PERSONAS = {
        "commuter": "서울에서 출퇴근하는 직장인. 점심시간과 퇴근길에 소비하며, 효율적이고 합리적인 소비를 추구함.",
        "weekend_visitor": "주말에 서울 상권을 방문하는 소비자. 여가와 쇼핑을 즐기며, 새로운 경험을 좋아함.",
        "resident": "서울에 거주하며 동네 상권을 이용하는 소비자. 단골 가게를 선호하며, 동네 변화에 민감함.",
        "evening_visitor": "저녁/야간에 외출하여 소비하는 사람. 술자리, 야식, 회식 등 저녁 활동 중심.",
    }

    news = env_summary.get("news", [])
    policy = env_summary.get("active_policy", "없음")

    # Ollama 사용 가능 여부 확인
    llm_available = False
    try:
        from llm_client import OLLAMA_URL, MODEL_NAME
        r = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=3)
        llm_available = r.status_code == 200
    except Exception:
        pass

    for aid, summ in selected:
        agent = next((a for a in agents if a["agent_id"] == aid), None)
        if not agent:
            continue

        seg = summ["segment"]
        seg_kr = SEG_NAMES.get(seg, seg)
        age = agent.get("age_group", "?")
        gender = agent.get("gender", "?")
        loyalty = agent.get("loyalty", 0.5)
        trend_sensitivity = agent.get("trend_sensitivity", 0.5)
        top_ind = summ.get("top_industry", "다양한 업종")
        sat = summ.get("satisfaction", 0.5)
        triggers = summ.get("triggers", {})
        new_explores = summ.get("new_explores", [])
        all_inds = summ.get("all_industries", [])
        spending = summ.get("spending", 0)

        # 지난주 대비 변화
        spend_delta = ""
        if prev_report:
            prev_summ = prev_report.get("_agent_summaries", {}).get(aid, {})
            prev_spending = prev_summ.get("spending", 0)
            if prev_spending > 0:
                pct = (spending - prev_spending) / prev_spending * 100
                if pct > 15:
                    spend_delta = f"지난주 대비 {pct:.0f}% 증가"
                elif pct < -15:
                    spend_delta = f"지난주 대비 {abs(pct):.0f}% 감소"
                else:
                    spend_delta = "지난주와 비슷"

        # ── LLM 기반 인터뷰 생성 ──
        if llm_available:
            comment = _llm_interview(
                agent, summ, seg, SEG_PERSONAS.get(seg, ""),
                news, policy, spend_delta, new_explores, triggers, sat,
            )
        else:
            comment = _persona_interview(
                agent, summ, seg, age, gender, loyalty, trend_sensitivity,
                top_ind, sat, triggers, new_explores, all_inds,
                spending, spend_delta, news, policy,
            )

        interviews.append({
            "agent_id": aid,
            "segment": seg_kr,
            "age": age,
            "gender": gender,
            "spending": spending,
            "satisfaction": sat,
            "triggers": list(triggers.keys())[:3],
            "new_explores": new_explores,
            "comment": comment,
        })

    return interviews


def _llm_interview(agent, summ, seg, persona_desc,
                   news, policy, spend_delta, new_explores, triggers, sat):
    """LLM (Ollama)으로 개인화된 인터뷰 코멘트 생성."""
    import requests, json
    from llm_client import OLLAMA_URL, MODEL_NAME

    age = agent.get("age_group", "?")
    gender = agent.get("gender", "?")
    loyalty = agent.get("loyalty", 0.5)
    trend_sens = agent.get("trend_sensitivity", 0.5)
    top_ind = summ.get("top_industry", "?")

    trigger_list = ", ".join(list(triggers.keys())[:3]) if triggers else "특별한 건 없었음"
    explore_list = ", ".join(new_explores[:3]) if new_explores else "없음"
    news_text = " / ".join(news[:2]) if news else "특이사항 없음"

    loyalty_desc = "단골 성향이 강함" if loyalty > 0.6 else "새로운 곳 잘 감" if loyalty < 0.4 else "보통"
    trend_desc = "트렌드에 민감함" if trend_sens > 0.6 else "유행에 관심 없음" if trend_sens < 0.4 else "보통"

    sat_desc = ("매우 불만족" if sat < 0.3 else
                "약간 불만족" if sat < 0.45 else
                "보통" if sat < 0.55 else
                "만족" if sat < 0.75 else "매우 만족")

    prompt = f"""당신은 서울의 소비자입니다. 아래 프로필을 기반으로 이번 주 소비 경험에 대해 1인칭으로 솔직하게 이야기하세요.

[프로필]
- {age} {gender}, {persona_desc}
- 소비 성향: {loyalty_desc}, {trend_desc}
- 주로 방문하는 업종: {top_ind}

[이번 주 경험]
- 방문한 업종: {', '.join(summ.get('all_industries', []))}
- 처음 시도해본 업종: {explore_list}
- 총 소비: {summ.get('spending', 0):,}원 ({spend_delta or '첫 주'})
- 만족도: {sat_desc} ({sat:.2f})
- 영향 받은 요인: {trigger_list}
- 주요 뉴스: {news_text}
- 정책: {policy}

[지시]
- 2~3문장으로 답변하세요.
- 본인의 나이, 성별, 성격에 맞는 말투를 사용하세요.
- 구체적인 감정과 생각을 표현하세요 (좋았다/아쉬웠다/화났다/신기했다 등).
- 템플릿 같은 문장은 절대 사용하지 마세요.
- /no_think"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 200},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # think 태그 제거
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # 너무 길면 자르기
        lines = raw.split("\n")
        result = " ".join(l.strip() for l in lines if l.strip())
        if len(result) > 300:
            result = result[:297] + "..."
        if result:  # LLM이 빈 응답을 반환한 경우 폴백
            return result

    except Exception:
        pass

    # 폴백: LLM 실패 또는 빈 응답
    return _persona_interview(
        agent, summ, summ["segment"],
        agent.get("age_group", "?"), agent.get("gender", "?"),
        agent.get("loyalty", 0.5), agent.get("trend_sensitivity", 0.5),
        summ.get("top_industry", "?"), summ.get("satisfaction", 0.5),
        summ.get("triggers", {}), summ.get("new_explores", []),
        summ.get("all_industries", []),
        summ.get("spending", 0), "", [], "없음",
    )


def _persona_interview(agent, summ, seg, age, gender, loyalty, trend_sensitivity,
                       top_ind, sat, triggers, new_explores, all_inds,
                       spending, spend_delta, news, policy):
    """LLM 없이 페르소나 기반 다양한 코멘트 생성 (폴백).

    나이/성별/성향별 자연스러운 완결 문장을 조합.
    """
    import hashlib
    h = int(hashlib.md5(agent["agent_id"].encode()).hexdigest()[:8], 16)

    parts = []

    # ── 나이 기반 톤 ──
    age_num = int(age.split("_")[0]) if "_" in age else 30
    if age_num < 25:
        tone = "casual"
    elif age_num < 40:
        tone = "polite"
    elif age_num < 55:
        tone = "mature"
    else:
        tone = "elderly"

    # ── 만족도 기반 감정 (톤별 완결 문장) ──
    SAT_COMMENTS = {
        "low": {  # sat < 0.3
            "casual": [
                "이번 주 진짜 별로였어요... 갈 데가 없었달까",
                f"돈만 쓰고 마음에 드는 데가 없었어요. {top_ind}도 좀 실망이었고",
                "아 진짜 어디를 가야 할지 모르겠어요. 다 거기서 거기",
            ],
            "polite": [
                "이번 주는 아쉬운 점이 많았어요. 마음에 드는 곳을 못 찾았거든요.",
                f"솔직히 {top_ind} 가격 대비 만족도가 낮았어요.",
                "지출은 했는데 돌아보면 좀 허전한 한 주였어요.",
            ],
            "mature": [
                "이번 주는 전반적으로 불만족스러웠습니다. 갈 만한 곳이 점점 줄어드는 느낌이에요.",
                f"{top_ind}을 몇 번 갔는데, 예전만 못하더라고요.",
                "소비 자체가 즐겁지 않은 주였습니다.",
            ],
            "elderly": [
                "요즘 가게들이 영 마음에 안 들어. 옛날이 나았지.",
                f"{top_ind} 어딜 가도 비싸기만 하고 맛은 그냥 그래.",
                "이번 주는 좀 실망스러웠어. 뭘 해도 마음에 안 차더라고.",
            ],
        },
        "mid_low": {  # 0.3 <= sat < 0.45
            "casual": [
                "그냥 무난했어요. 특별한 건 없었고요.",
                f"뭐... {top_ind} 먹고 그냥 왔어요. 딱히 감흥은 없었네요",
                "좋지도 나쁘지도 않은? 뭔가 좀 아쉬운 한 주였어요.",
            ],
            "polite": [
                "보통의 한 주였어요. 기대한 만큼은 아니었지만 나쁘진 않았어요.",
                f"{top_ind} 위주로 다녔는데, 만족도가 좀 들쑥날쑥했어요.",
                "가끔 '왜 여기 왔지' 싶을 때가 있었어요.",
            ],
            "mature": [
                "무난하게 지냈습니다. 크게 좋지도 나쁘지도 않았어요.",
                f"습관적으로 {top_ind}을 가긴 했는데, 만족감은 좀 부족하더군요.",
                "이번 주는 소비 자체에 큰 감흥이 없었습니다.",
            ],
            "elderly": [
                "뭐 이번 주도 그냥저냥 보냈지. 별 거 없었어.",
                f"맨날 가던 {top_ind}이나 갔지. 다른 데는 귀찮아서.",
                "특별할 것 없는 한 주였어. 조용히 지냈지 뭐.",
            ],
        },
        "mid": {  # 0.45 <= sat < 0.6
            "casual": [
                f"{top_ind} 가서 먹고 좀 돌아다녔어요. 나쁘지 않았어요!",
                "평소처럼 보냈어요. 편하고 좋았어요.",
                "딱 이 정도면 괜찮은 한 주였어요.",
            ],
            "polite": [
                f"이번 주는 {top_ind} 위주로 편하게 다녔어요.",
                "큰 변화 없이 평소 루틴대로 소비했어요.",
                "무난하게 잘 보낸 한 주였어요.",
            ],
            "mature": [
                f"평소처럼 {top_ind} 중심으로 지냈습니다.",
                "이번 주도 무탈하게 보냈습니다. 편안한 주였어요.",
                "일상적인 소비 패턴이었습니다. 나쁘지 않았어요.",
            ],
            "elderly": [
                f"늘 가던 {top_ind}이 편하지. 이번 주도 괜찮았어.",
                "평소대로 동네에서 이것저것 다녀왔어. 나쁘지 않았지.",
                "잔잔하게 보냈어. 이 정도면 만족해.",
            ],
        },
        "high": {  # 0.6 <= sat < 0.75
            "casual": [
                f"이번 주 {top_ind} 진짜 좋았어요! 또 가고 싶어요.",
                "우와 이번 주 꽤 괜찮은 발견을 했어요!",
                "기대 이상이었어요! 기분 좋은 한 주였어요.",
            ],
            "polite": [
                f"{top_ind}이 기대 이상이었어요. 다음에 또 가려고요.",
                "전반적으로 만족스러운 소비를 한 것 같아요.",
                "이번 주는 꽤 괜찮은 경험을 했어요.",
            ],
            "mature": [
                f"이번 주 {top_ind}이 특히 좋았습니다. 마음에 드는 곳을 찾았네요.",
                "만족스러운 한 주를 보냈습니다.",
                "소비한 만큼의 가치가 충분히 있었어요.",
            ],
            "elderly": [
                f"{top_ind} 괜찮았어. 오랜만에 마음에 드는 데를 찾았지.",
                "이번 주는 좋았어. 기분 좋게 다녀왔어.",
                "나쁘지 않았어! 이런 주가 계속됐으면 좋겠어.",
            ],
        },
        "very_high": {  # sat >= 0.75
            "casual": [
                f"대박! 이번 주 {top_ind} 완전 인생이었어요!!",
                "정말 좋은 한 주였어요! 돈이 아깝지 않았어요.",
                f"완전 만족! {top_ind} 최고였어요 ㅎㅎ",
            ],
            "polite": [
                f"이번 주 정말 만족스러웠어요! {top_ind}이 특히 인상적이었어요.",
                "오랜만에 이렇게 즐거운 소비를 한 것 같아요.",
                "모든 게 마음에 들었어요. 완벽한 한 주였어요.",
            ],
            "mature": [
                f"이번 주는 정말 만족스럽습니다. {top_ind}이 기대 이상이었어요.",
                "오랜만에 소비의 즐거움을 느꼈습니다.",
                "아주 좋은 경험을 했습니다. 추천하고 싶을 정도예요.",
            ],
            "elderly": [
                f"오랜만에 진짜 좋았어. {top_ind}이 참 맛있더라고.",
                "이런 날이 있으니까 사는 맛이 나지. 완전 만족!",
                "아주 좋은 한 주였어. 기분이 좋아.",
            ],
        },
    }

    # 만족도 구간 결정
    if sat < 0.3:
        sat_key = "low"
    elif sat < 0.45:
        sat_key = "mid_low"
    elif sat < 0.6:
        sat_key = "mid"
    elif sat < 0.75:
        sat_key = "high"
    else:
        sat_key = "very_high"

    sat_options = SAT_COMMENTS[sat_key][tone]
    parts.append(sat_options[h % len(sat_options)])

    # ── 소비 성향 코멘트 ──
    LOYALTY_COMMENTS = {
        "casual": [
            f"역시 자주 가는 {top_ind}이 편해요.",
            f"단골이니까 믿고 가는 {top_ind}!",
        ],
        "polite": [
            f"자주 가는 {top_ind}이라 편안했어요.",
            f"단골 가게가 있어서 다행이에요. {top_ind}은 항상 좋아요.",
        ],
        "mature": [
            f"역시 단골인 {top_ind}이 편안하더군요.",
            f"오랫동안 다닌 {top_ind}이 믿음직합니다.",
        ],
        "elderly": [
            f"역시 가던 {top_ind}이 편하지.",
            f"단골이니까. {top_ind}은 변함없이 좋아.",
        ],
    }

    EXPLORE_COMMENTS = {
        "casual": [
            "{txt} 처음 가봤는데 신선했어요!",
            "새로운 데 가보는 게 재밌어요. {txt}이 좋았어요.",
        ],
        "polite": [
            "{txt}을 처음 시도해봤는데, 생각보다 괜찮았어요.",
            "이번에 {txt}을 새로 경험했는데, 좋은 발견이었어요.",
        ],
        "mature": [
            "{txt}을 처음 가봤습니다. 나쁘지 않더군요.",
            "새로 {txt}을 가봤는데, 의외로 만족스러웠어요.",
        ],
        "elderly": [
            "{txt}을 가봤는데, 요즘 것도 괜찮더라고.",
            "새 가게 {txt}을 가봤어. 나쁘지 않았어.",
        ],
    }

    if loyalty > 0.7:
        opts = LOYALTY_COMMENTS[tone]
        parts.append(opts[h % len(opts)])
    elif loyalty < 0.3 and new_explores:
        txt = ", ".join(new_explores[:2])
        opts = EXPLORE_COMMENTS[tone]
        parts.append(opts[h % len(opts)].format(txt=txt))

    # ── 트렌드 민감도 ──
    if trend_sensitivity > 0.7 and news:
        news_short = news[0][:25] if news else ""
        TREND_COMMENTS = {
            "casual": f"'{news_short}' 보고 관심 생겼어요!",
            "polite": f"'{news_short}' 기사를 보고 한번 가보고 싶었어요.",
            "mature": f"'{news_short}' 관련 뉴스를 보고 관심이 생겼습니다.",
            "elderly": f"'{news_short}' 소식 듣고 가봤어.",
        }
        parts.append(TREND_COMMENTS[tone])

    # ── 소비 변화 ──
    if spend_delta:
        if "증가" in spend_delta:
            SPEND_UP = {
                "casual": "이번 주 좀 많이 썼는데... 뭐 맛있었으니까요!",
                "polite": "소비가 좀 늘었지만, 그만큼 좋은 경험이었어요.",
                "mature": "지출이 늘긴 했는데, 나름 가치 있는 소비였습니다.",
                "elderly": "좀 많이 쓰긴 했지. 뭐 가끔은 그럴 수도 있고.",
            }
            parts.append(SPEND_UP[tone])
        elif "감소" in spend_delta:
            SPEND_DN = {
                "casual": "이번 주는 좀 줄였어요. 지난주에 너무 써서 ㅋㅋ",
                "polite": "이번 주는 약간 절약 모드였어요.",
                "mature": "절제 모드로 지냈습니다. 소비를 좀 줄였어요.",
                "elderly": "이번 주는 좀 아꼈지. 안 써도 되는 건 안 쓰고.",
            }
            parts.append(SPEND_DN[tone])

    # ── 정책 반응 ──
    if policy and policy != "없음":
        POLICY_COMMENTS = {
            "casual": f"{policy} 덕에 좀 도움 됐어요!",
            "polite": f"{policy} 덕분에 부담이 좀 줄었어요.",
            "mature": f"{policy}가 소비에 도움이 됐습니다.",
            "elderly": f"정부에서 {policy} 해주니까 좋더라고.",
        }
        parts.append(POLICY_COMMENTS[tone])

    comment = " ".join(parts)
    if len(comment) > 300:
        comment = comment[:297] + "..."
    return comment


def _save_json(report, week):
    """JSON 저장."""
    path = REPORTS_DIR / f"week_{week:02d}.json"
    # set은 JSON 직렬화 불가 → 제거
    clean = json.loads(json.dumps(report, default=str, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def _save_markdown(report, week):
    """Markdown 보고서 저장."""
    path = REPORTS_DIR / f"week_{week:02d}.md"
    env = report.get("env_summary", {})

    lines = [
        f"# Week {week} 소비행동 보고서",
        "",
        "---",
        "",
        "## 개요",
        f"- 에이전트: {report.get('agent_count', 0)}명",
        f"- 총 소비: {report.get('total_spending', 0):,}원",
        f"- 에이전트당 평균: {report.get('avg_spending_per_agent', 0):,}원",
        f"- 총 소비 행동: {report.get('total_actions', 0)}건",
        "",
        "## 환경 상태",
        f"- 정책: {env.get('active_policy', '없음')}",
        f"- 유동인구 변화: {env.get('pop_change', '0%')}",
    ]

    news = env.get("news", [])
    if news:
        lines.append(f"- 뉴스: {', '.join(news)}")

    # 정책 이벤트
    pol_events = env.get("policy_events", [])
    if pol_events:
        lines.append("")
        lines.append("### 정책 이벤트")
        for evt in pol_events:
            lines.append(f"- {evt.get('msg', str(evt))}")

    # 상권 변화
    dist_events = env.get("district_events", [])
    if dist_events:
        lines.append("")
        lines.append("### 상권 변화")
        for evt in dist_events[:8]:
            lines.append(f"- {evt.get('msg', str(evt))}")

    # 세그먼트별
    lines.append("")
    lines.append("## 세그먼트별 행동 분석")
    lines.append("")

    SEG_ORDER = ["commuter", "weekend_visitor", "resident", "evening_visitor"]
    SEG_NAMES = {
        "commuter": "출퇴근 직장인",
        "weekend_visitor": "주말 방문객",
        "resident": "지역 거주민",
        "evening_visitor": "저녁/야간 방문",
    }

    for seg in SEG_ORDER:
        sr = report["segment_reports"].get(seg)
        if not sr:
            continue
        name = SEG_NAMES.get(seg, seg)
        lines.append(f"### {name} ({sr['agent_count']}명)")
        lines.append(f"- 평균 소비: {sr['avg_spending']:,}원")
        lines.append(f"- 평균 행동: {sr['avg_actions_per_agent']}건/주")
        lines.append(f"- 평균 만족도: {sr['avg_satisfaction']}")
        lines.append(f"- TOP 업종: {', '.join(list(sr['top_industries'].keys())[:3])}")

        # 행동 분포
        bd = sr.get("behavior_distribution", {})
        if bd:
            parts = [f"{k} {v}%" for k, v in bd.items()]
            lines.append(f"- 행동: {' / '.join(parts)}")

        # 전주 대비
        delta = report["deltas"].get(seg)
        if delta:
            sp_d = delta["spending_delta"]
            sat_d = delta["satisfaction_delta"]
            act_d = delta["actions_delta"]
            lines.append(f"- 변화: 소비 {sp_d:+,}원, 만족도 {sat_d:+.2f}, 행동 {act_d:+.1f}건")

        lines.append("")

    # 전후 행동 패턴 비교
    bchanges = report.get("behavior_changes", {})
    if bchanges:
        lines.append("## 소비행동 패턴 전후 비교")
        lines.append("")
        for seg in SEG_ORDER:
            bc = bchanges.get(seg)
            if not bc:
                continue
            name = SEG_NAMES.get(seg, seg)
            lines.append(f"### {name}")

            spend_t = bc.get("spending_trend", "유지")
            sat_t = bc.get("satisfaction_trend", "유지")
            lines.append(f"- 소비 트렌드: **{spend_t}** / 만족도 트렌드: **{sat_t}**")

            new_ind = bc.get("new_top_industries", [])
            drop_ind = bc.get("dropped_top_industries", [])
            if new_ind:
                lines.append(f"- 새로 떠오른 업종: {', '.join(new_ind)}")
            if drop_ind:
                lines.append(f"- 순위 하락 업종: {', '.join(drop_ind)}")

            shifts = bc.get("behavior_shifts", {})
            if shifts:
                shift_parts = []
                for btype, delta in shifts.items():
                    arrow = "↑" if delta > 0 else "↓"
                    shift_parts.append(f"{btype} {arrow}{abs(delta)}%p")
                lines.append(f"- 행동 변화: {', '.join(shift_parts)}")
            lines.append("")

    # 인터뷰 — 세그먼트별 그룹핑
    interviews = report.get("interviews", [])
    if interviews:
        lines.append("## 에이전트 인터뷰")
        lines.append("")

        # 이번 주 이벤트 요약 (인터뷰 컨텍스트)
        lines.append("### 이번 주 이벤트/환경")
        if news:
            for n in news:
                lines.append(f"- 📢 {n}")
        if pol_events:
            for evt in pol_events:
                lines.append(f"- 📋 {evt.get('msg', str(evt))}")
        dist_summary = env.get("district_events", [])
        openings = [e for e in dist_summary if e.get("type") == "NEW_STORE"]
        closings = [e for e in dist_summary if e.get("type") == "STORE_CLOSED"]
        if openings:
            lines.append(f"- 🏪 신규 입점: {len(openings)}건")
        if closings:
            lines.append(f"- 🚫 폐업: {len(closings)}건")
        if not news and not pol_events and not openings and not closings:
            lines.append("- 특별한 이벤트 없음")
        lines.append("")

        # 세그먼트별 그룹핑
        seg_groups = defaultdict(list)
        for iv in interviews:
            seg_groups[iv["segment"]].append(iv)

        for seg_name, ivs in seg_groups.items():
            lines.append(f"### {seg_name} 인터뷰 ({len(ivs)}명)")
            lines.append("")
            for iv in ivs:
                triggers_str = ""
                if iv.get("triggers"):
                    triggers_str = f" | 영향: {', '.join(iv['triggers'][:2])}"
                lines.append(
                    f"**{iv['agent_id']}** ({iv['age']} {iv['gender']}){triggers_str}"
                )
                lines.append(f"> \"{iv['comment']}\"")
                lines.append(f"> 소비: {iv['spending']:,}원 / 만족도: {iv['satisfaction']}")
                lines.append("")

    # 커뮤니티 분석
    communities = report.get("community_analysis", {})
    if communities:
        lines.append("## 소셜 네트워크 커뮤니티 분석")
        lines.append("")

        SEG_KR = {
            "commuter": "직장인", "weekend_visitor": "주말방문객",
            "resident": "거주민", "evening_visitor": "저녁방문자",
        }

        for cid, ca in sorted(communities.items(),
                              key=lambda x: x[1]["members"], reverse=True):
            seg_kr = SEG_KR.get(ca['dominant_segment'], ca['dominant_segment'])
            top_inds = ', '.join(list(ca['top_industries'].keys())[:3])
            lines.append(f"### 커뮤니티 {cid} ({ca['members']}명, 주로 {seg_kr})")
            lines.append(f"- 평균 소비: {ca['avg_spending']:,}원")
            lines.append(f"- 평균 만족도: {ca['avg_satisfaction']}")
            lines.append(f"- 상위 업종: {top_inds}")
            interp = ca.get("interpretation", "")
            if interp:
                lines.append(f"> 💡 {interp}")
            lines.append("")


    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [Report] {path.name}")


# ═══════════════════════════════════════════

if __name__ == "__main__":
    # 테스트
    test_agents = [
        {"agent_id": f"consumer_{i:04d}", "segment": s, "adm_cd": "1114055000",
         "home_adm_cd": "1121560000", "age_group": "30_39세", "gender": "남"}
        for i, s in enumerate(["commuter"] * 4 + ["weekend_visitor"] * 2 + ["resident"] * 2)
    ]
    test_logs = {}
    rng = np.random.default_rng(42)
    for a in test_agents:
        week_data = []
        for d in range(7):
            day_actions = [{
                "activity": "lunch", "industry": rng.choice(["한식", "카페", "중식"]),
                "time_slot": 12, "amount": int(rng.integers(8000, 25000)),
                "dong": "11140550", "satisfaction": round(float(rng.random()), 2),
                "type": "외출_소비",
            }]
            week_data.append(day_actions)
        test_logs[a["agent_id"]] = week_data

    env_summary = {
        "active_policy": "소상공인 쿠폰",
        "pop_change": "+5.0%",
        "news": ["을지로 맛집 특집"],
        "policy_events": [],
        "district_events": [],
    }

    report = generate_weekly_report(0, test_agents, test_logs, env_summary)
    print(f"\n[OK] Report generated: week={report['week']}, "
          f"spending={report.get('total_spending', 0):,}")
