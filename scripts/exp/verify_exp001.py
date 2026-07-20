# -*- coding: utf-8 -*-
"""EXP-001 종합 검증(상세판) — 속도·품질·추출완전성·소비행동·이동·정책·사회·무결성."""
import json, glob, os, re, sys
from datetime import date

NAS = "/home/ubuntu/data/exp001"
METRICS = f"{NAS}/sim_output/metrics"
RUNLOG = f"{NAS}/logs/run.log"
SIM_START = date(2025, 7, 14)
POLICY_FROM = date(2025, 7, 21)
GRANT = {1: 400000, 2: 300000, 3: 150000, 4: 150000, 5: 150000,
         6: 150000, 7: 150000, 8: 150000, 9: 150000, 10: 150000}
WD = ["월", "화", "수", "목", "금", "토", "일"]
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "exp001pass")
sys.path.insert(0, "/data/exp001_repo/scripts/neo4j_load")


def hr(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


def logtext():
    return open(RUNLOG, encoding="utf-8", errors="ignore").read() if os.path.exists(RUNLOG) else ""


def load_days():
    days = {}
    for f in sorted(glob.glob(f"{METRICS}/day_*.jsonl")):
        d = f.split("day_")[1].replace(".jsonl", "")
        rows = []
        for l in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
        days[d] = rows
    return days


def wlabel(d):
    return WD[date.fromisoformat(d).weekday()] + ("·주말" if date.fromisoformat(d).weekday() >= 5 else "")


def latest_day(days):
    return max(days) if days else None


def completed(days):
    return [d for d in sorted(days) if sum(1 for r in days[d] if r.get("status") == "ok") >= 7000]


def A_speed(days):
    hr("A. 속도 / 진행")
    # resume 시 같은 날 'done in' 이 여러 번(스킵 42s + 실제 처리) 찍힌다.
    # 각 날의 MAX 소요를 실제 처리시간으로 채택(스킵 시간 무시).
    dt = {}
    for d, sec in re.findall(r"\[Day \d+ ([\d-]+)\] done in (\d+)s", logtext()):
        dt[d] = max(dt.get(d, 0), int(sec))
    tot = 0
    for d in sorted(days):
        rows = days[d]
        ok = sum(1 for r in rows if r.get("status") == "ok")
        err = len(rows) - ok
        tot += ok
        dur = dt.get(d)
        complete = ok >= 7000
        rate = f"{ok/(dur/60):.1f}/min" if dur and dur > 60 and complete else ("진행중" if not complete else "-")
        done = f"{dur//60}m{dur%60}s" if dur and dur > 60 and complete else ("…미완" if not complete else "skip")
        print(f"  {d}({wlabel(d):>5}): {ok:>4}ok/{err}err  {done:>10}  {rate:>9}")
    print(f"\n  누적 {tot:,} agent-day / {len(days)}/14 일 착수")
    # 현재 진행 중인 날 = 미완(<7000) 최신일. metrics 실측 카운트로 표시(로그 stale 회피)
    cur = next((d for d in sorted(days) if sum(1 for r in days[d] if r.get("status") == "ok") < 7000), None)
    if cur:
        n = sum(1 for r in days[cur] if r.get("status") == "ok")
        print(f"  현재 처리 중: {cur}({wlabel(cur)}) {n}/7500 (metrics 실측)")
    # ETA: 완주한 날들의 실제 소요만 (스킵 제외)
    durs = [v for d, v in dt.items() if v > 60 and sum(1 for r in days.get(d, []) if r.get("status") == "ok") >= 7000]
    ncomp = sum(1 for d in days if sum(1 for r in days[d] if r.get("status") == "ok") >= 7000)
    if durs:
        avg = sum(durs) / len(durs)
        remain = 14 - ncomp
        print(f"  실처리일 평균 {avg/60:.0f}분(주말↑) -> 잔여 ~{remain}일 예상 {avg*remain/3600:.1f}h")


def B_quality(days):
    hr("B. 품질 / 오류")
    allr = [r for rows in days.values() for r in rows]
    ok = [r for r in allr if r.get("status") == "ok"]
    if not ok:
        print("  완료 agent 없음")
        return
    av = lambda k: sum((r.get(k, 0) or 0) for r in ok) / len(ok)
    errs = [r for r in allr if r.get("status") != "ok"]
    print(f"  ok={len(ok):,} err={len(errs)} err율={100*len(errs)/len(allr):.3f}%")
    print(f"  s1_attempts={av('s1_attempts'):.2f} s2_attempts={av('s2_attempts'):.2f} (1=JSON 정상)")
    print(f"  tokens_out avg={av('tokens_out'):.0f} tokens_in avg={av('tokens_in'):.0f}")
    print(f"  환각 교정={sum(r.get('fb_hallucinations_corrected',0) for r in ok)} "
          f"드롭={sum(r.get('fb_hallucinations_dropped',0) for r in ok)} "
          f"후보전무={sum(r.get('fb_cand_all_empty',0) for r in ok)}")
    if errs:
        print("  [오류 상세]")
        for r in errs[:5]:
            e = str(r.get("error", ""))[:90]
            print(f"    {r.get('aid')}: {e}")


def C_completeness(days):
    hr("C. 추출 완전성 (의도 정보가 값으로 채워졌는가)")
    d = latest_day(days)
    if not d:
        print("  데이터 없음")
        return
    from _common import driver_session
    with driver_session() as s:
        r = s.run("""MATCH (:Plan {day:date($d)})-[i:INCLUDES]->(poi:POI)
          RETURN count(i) AS n,
            sum(CASE WHEN i.actual_spent IS NOT NULL THEN 1 ELSE 0 END) AS spent,
            sum(CASE WHEN i.spent_from_policy IS NOT NULL THEN 1 ELSE 0 END) AS pol,
            sum(CASE WHEN i.category IS NOT NULL THEN 1 ELSE 0 END) AS cat,
            sum(CASE WHEN i.actual_spent>0 THEN 1 ELSE 0 END) AS spos,
            sum(CASE WHEN i.actual_satisfaction IS NOT NULL THEN 1 ELSE 0 END) AS sat""", d=d).single()
        n = r["n"] or 1
        print(f"  [INCLUDES {r['n']}건 @ {d}] actual_spent {100*r['spent']/n:.0f}% · "
              f"spent_from_policy {100*r['pol']/n:.0f}% · category {100*r['cat']/n:.0f}% · "
              f"만족도 {r['sat']}/{r['spos']}(소비이벤트만=정상)")
        r = s.run("""MATCH (a:Agent)-[:HAS_STATE {day:date($d)}]->(st:State)
          RETURN count(st) AS n,
            sum(CASE WHEN st.grant_received IS NOT NULL THEN 1 ELSE 0 END) AS gr,
            sum(CASE WHEN st.grant_remaining IS NOT NULL THEN 1 ELSE 0 END) AS grem,
            sum(CASE WHEN st.policy_used IS NOT NULL THEN 1 ELSE 0 END) AS pu""", d=d).single()
        n = r["n"] or 1
        print(f"  [State {r['n']}개] 지갑 grant_received {100*r['gr']/n:.0f}% · "
              f"grant_remaining {100*r['grem']/n:.0f}% · policy_used {100*r['pu']/n:.0f}%")


def D_consumption(days):
    hr("D. 소비 행동 상세")
    from _common import driver_session
    comp = completed(days)
    with driver_session() as s:
        print("  [일별]  요일   외출율  1인소비   만족도  mood  fatigue")
        for d in comp:
            rows = [r for r in days[d] if r.get("status") == "ok"]
            outing = 100 * sum(1 for r in rows if (r.get("n_events", 0) or 0) > 0) / len(rows)
            sat = sum((r.get("avg_sat", 0) or 0) for r in rows) / len(rows)
            mood = sum((r.get("mood", 0) or 0) for r in rows) / len(rows)
            fat = sum((r.get("fatigue", 0) or 0) for r in rows) / len(rows)
            sp = s.run("""MATCH (a:Agent)-[:HAS_PLAN {day:date($d)}]->(p)
              OPTIONAL MATCH (p)-[i:INCLUDES]->() WITH a, sum(coalesce(i.actual_spent,0)) AS x
              RETURN avg(x) AS avg""", d=d).single()["avg"]
            print(f"  {d}({wlabel(d):>5}) {outing:5.1f}% {int(sp or 0):>8,} {sat:6.2f} {mood:5.2f} {fat:6.2f}")
        d = comp[-1] if comp else latest_day(days)
        if d:
            # 소비 위치 정합: 소비는 외부 상권(zone)에서 일어나야 정상 (거주/직장 앵커 아님)
            loc = s.run("""MATCH (:Plan {day:date($d)})-[i:INCLUDES]->() WHERE i.actual_spent>0
              RETURN count(i) AS n,
                sum(CASE WHEN i.anchor IN ['residence','workplace'] THEN 1 ELSE 0 END) AS home_work""", d=d).single()
            if loc["n"]:
                print(f"\n  [소비 위치 정합 @ {d}] 소비 {loc['n']}건 中 거주/직장 앵커 {loc['home_work']}건 "
                      f"({100*loc['home_work']/loc['n']:.2f}%) — 0%에 가까워야 정상(상권서 소비)")
            r = s.run("""MATCH (a:Agent)-[:HAS_PLAN {day:date($d)}]->(p)
              OPTIONAL MATCH (p)-[i:INCLUDES]->() WITH a, sum(coalesce(i.actual_spent,0)) AS sp
              RETURN percentileCont(sp,0.5) AS med, percentileCont(sp,0.9) AS p90,
                percentileCont(sp,0.99) AS p99, max(sp) AS mx,
                sum(CASE WHEN sp>=300000 THEN 1 ELSE 0 END) AS over,
                sum(CASE WHEN sp=0 THEN 1 ELSE 0 END) AS zero, count(a) AS n""", d=d).single()
            n = r["n"] or 1
            print(f"\n  [소비 분포 @ {d}] 중앙 {int(r['med']):,} · p90 {int(r['p90']):,} · "
                  f"p99 {int(r['p99']):,} · 최대 {int(r['mx']):,}")
            print(f"     하루 30만원↑ {r['over']}명({100*r['over']/n:.1f}%, 과소비 추적) · "
                  f"0원 {r['zero']}명({100*r['zero']/n:.1f}%)")
            print("  [소비 10분위별 1인 소비]")
            deciles = list(s.run("""MATCH (a:Agent)-[:HAS_PLAN {day:date($d)}]->(p)
              OPTIONAL MATCH (p)-[i:INCLUDES]->() WITH a.spending_level_wd AS dec, a, sum(coalesce(i.actual_spent,0)) AS sp
              RETURN dec, count(a) AS n, avg(sp) AS avg ORDER BY dec""", d=d))
            line = "   ".join(f"{r['dec']}분위 {int(r['avg'] or 0):,}" for r in deciles if r['dec'])
            print(f"     {line}")


def E_mobility(days):
    hr("E. 이동 / 직장 외출 / 소비 지리")
    from _common import driver_session
    comp = completed(days)
    with driver_session() as s:
        nworker = s.run("MATCH (a:Agent)-[:WORKS_AT]->() RETURN count(DISTINCT a) AS c").single()["c"]
        print(f"  직장 보유 agent {nworker:,}명 기준 직장 방문율 (평일 높고 주말 낮아야 정상)")
        for d in comp:
            w = s.run("""MATCH (a:Agent)-[:WORKS_AT]->(wp)
              MATCH (a)-[:HAS_PLAN {day:date($d)}]->()-[:INCLUDES]->(wp)
              RETURN count(DISTINCT a) AS c""", d=d).single()["c"]
            print(f"  {d}({wlabel(d):>5}) 직장방문 {w:>4}명({100*w/nworker:4.1f}%)")
        # 소비 지리: 방문 POI가 거주동/거주구/타구인지 (주말 원거리 소비 ↑ 정상)
        print("\n  [소비 지리 — 방문 POI 위치] 거주同 / 거주區(동제외) / 타자치구")
        for d in comp:
            r = s.run("""MATCH (a:Agent)-[:HAS_PLAN {day:date($d)}]->(:Plan)-[i:INCLUDES]->(:POI)-[:IN_DONG]->(pd:Dong)
              WHERE i.actual_spent>0
              MATCH (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(hd:Dong)
              WITH i, pd, hd,
                CASE WHEN pd.code=hd.code THEN 'sd'
                     WHEN substring(toString(pd.code),0,5)=substring(toString(hd.code),0,5) THEN 'sg'
                     ELSE 'ot' END AS loc
              RETURN count(*) AS n, sum(CASE WHEN loc='sd' THEN 1 ELSE 0 END) AS sd,
                sum(CASE WHEN loc='sg' THEN 1 ELSE 0 END) AS sg,
                sum(CASE WHEN loc='ot' THEN 1 ELSE 0 END) AS ot""", d=d).single()
            n = r["n"] or 1
            print(f"  {d}({wlabel(d):>5}) 거주同 {100*r['sd']/n:4.1f}% · 거주區 {100*r['sg']/n:4.1f}% · "
                  f"타구 {100*r['ot']/n:4.1f}% (거주區계 {100*(r['sd']+r['sg'])/n:.1f}%)")
        # 직장인 점심(11~14시) 소비 위치 — 최신 평일 (직장 근처 점심 재현 확인)
        wkdays = [d for d in comp if date.fromisoformat(d).weekday() < 5]
        if wkdays:
            d = wkdays[-1]
            r = s.run("""MATCH (a:Agent)-[:WORKS_AT]->(:POI)-[:IN_DONG]->(wd:Dong)
              MATCH (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(hd:Dong)
              MATCH (a)-[:HAS_PLAN {day:date($d)}]->(:Plan)-[i:INCLUDES]->(:POI)-[:IN_DONG]->(pd:Dong)
              WHERE i.actual_spent>0 AND i.time >= time('11:00:00') AND i.time <= time('14:00:00')
              RETURN count(*) AS n, sum(CASE WHEN pd.code=wd.code THEN 1 ELSE 0 END) AS work,
                sum(CASE WHEN pd.code=hd.code THEN 1 ELSE 0 END) AS home""", d=d).single()
            if r["n"]:
                print(f"\n  [직장인 점심(11~14시) 소비 @ {d}] 직장동 {100*r['work']/r['n']:.0f}% · "
                      f"거주동 {100*r['home']/r['n']:.0f}% (점심은 직장 근처가 정상)")


def F_policy(days):
    hr("F. 정책(P010) 의도 검증")
    from _common import driver_session
    active = [d for d in days if date.fromisoformat(d) >= POLICY_FROM]
    with driver_session() as s:
        if not active:
            cur = max((date.fromisoformat(d) for d in days), default=SIM_START)
            print(f"  시행일 {POLICY_FROM} 이전(현재 {cur}) -> 지급 없음=정상, {(POLICY_FROM-cur).days}일 남음")
            print("  [사전 예상] 시뮬 대상 분위별 인원 → 예상 지급 예산")
            rows = list(s.run("""MATCH (a:Agent)-[:HAS_PLAN]->() WITH DISTINCT a
              RETURN a.spending_level_wd AS dec, count(a) AS n ORDER BY dec"""))
            budget = 0
            parts = []
            for r in rows:
                dec = r["dec"]
                if dec is None:
                    continue
                g = GRANT.get(int(dec), 0)
                budget += r["n"] * g
                parts.append(f"{dec}:{r['n']}명×{g//10000}만")
            print("     " + "  ".join(parts))
            print(f"     예상 총 지급예산: {budget:,}원 (약 {budget/1e8:.1f}억)")
            cov = s.run("""MATCH (a:Agent)-[:HAS_PLAN]->() WITH DISTINCT a
              OPTIONAL MATCH (a)-[:KNOWS_POI]->(poi:POI)
              WITH a, count(poi) AS known, sum(CASE WHEN poi.coupon_eligible=true THEN 1 ELSE 0 END) AS cp
              RETURN avg(known) AS ak, avg(cp) AS ac,
                sum(CASE WHEN cp=0 THEN 1 ELSE 0 END) AS nocp, count(a) AS n""").single()
            print(f"  [쿠폰 사용처 준비] agent가 아는 POI 평균 {cov['ak']:.0f}개 중 쿠폰가맹 {cov['ac']:.0f}개 · "
                  f"쿠폰POI 0개 agent {cov['nocp']}명/{cov['n']} (0에 가까워야 지원금 소비 가능)")
        for d in sorted(active):
            rows = [r for r in days[d] if r.get("status") == "ok"]
            ap = [r for r in rows if r.get("grant_applied_today", 0) > 0]
            sp = [r for r in rows if r.get("policy_spend_today", 0) > 0]
            print(f"  {d}({wlabel(d):>5}): 신규지급 {len(ap)}명 · 쿠폰사용 {len(sp)}명 · "
                  f"당일사용 {sum(r.get('policy_spend_today',0) for r in rows):,}원 · "
                  f"잔여 {sum(r.get('grant_remaining_total',0) for r in rows):,}원")
        # 소진율: 최신 정책일 grant_received 대비 사용액 (지갑 State 실측)
        latest_pol = max(active) if active else None
        if latest_pol:
            grows = list(s.run("""MATCH (a:Agent)-[:HAS_STATE {day:date($d)}]->(st:State)
              WHERE st.grant_received CONTAINS 'P010'
              RETURN st.grant_received AS gr, st.grant_remaining AS grem""", d=latest_pol))
            tr = tu = 0
            for r in grows:
                try:
                    recv = json.loads(r["gr"]).get("P010", 0)
                    rem = json.loads(r["grem"]).get("P010", 0)
                    tr += recv; tu += (recv - rem)
                except Exception:
                    pass
            if tr:
                print(f"  [소진율 @ {latest_pol}] 수령 {tr:,}원 中 사용 {tu:,}원 → {100*tu/tr:.1f}% "
                      f"(수혜 {len(grows)}명)")
        rows = list(s.run("""MATCH (a:Agent)-[:HAS_STATE]->(st:State) WHERE st.grant_received CONTAINS 'P010'
          WITH a.spending_level_wd AS dec, st ORDER BY st.day DESC
          WITH dec, collect(st.grant_received)[0] AS gr
          RETURN dec, count(*) AS n, collect(gr)[0] AS smp ORDER BY dec"""))
        if rows:
            print("  [분위별 지급 정합]  분위 인원 1인지급 의도 판정")
            for r in rows:
                try:
                    amt = json.loads(r["smp"]).get("P010", 0)
                except Exception:
                    amt = 0
                exp = GRANT.get(int(r["dec"])) if r["dec"] else None
                v = "OK" if exp and amt == exp else "확인필요"
                print(f"    {str(r['dec']):>3} {r['n']:>5} {amt:>8,} {str(exp or '-'):>8}  {v}")
            # spent_from_policy는 JSON 문자열({"P010":금액}). 숫자 비교 불가 → CONTAINS로 판정
            rr = s.run("""MATCH (:Plan)-[i:INCLUDES]->(poi:POI)
              WHERE i.spent_from_policy IS NOT NULL AND i.spent_from_policy CONTAINS 'P010'
              RETURN count(i) AS n, sum(CASE WHEN poi.coupon_eligible=true THEN 1 ELSE 0 END) AS ok""").single()
            if rr["n"]:
                print(f"  [쿠폰 사용처 제한 준수] 지원금귀속 소비 {rr['n']}건 中 가맹점 {rr['ok']}건 "
                      f"({100*rr['ok']/rr['n']:.1f}%) — 100%여야 정상(비가맹 지원금 차단)")


def G_social(days):
    hr("G. 사회적 상호작용")
    from _common import driver_session
    comp = completed(days)
    with driver_session() as s:
        print("  [일별] 대화쌍 · 참여agent · intent(추천/약속/기타) · 소문")
        for d in comp:
            c = s.run("MATCH (c:Conversation {day:date($d)}) RETURN count(c) AS c", d=d).single()["c"]
            if c == 0:
                print(f"  {d}({wlabel(d):>5}) 야간 페이즈 대기중 (낮 완료·대화 미생성)")
                continue
            p = s.run("""MATCH (a:Agent)-[:PARTICIPATES_IN]->(c:Conversation {day:date($d)})
              RETURN count(DISTINCT a) AS c""", d=d).single()["c"]
            it = s.run("""MATCH (c:Conversation {day:date($d)})
              RETURN sum(CASE WHEN c.intent='추천' THEN 1 ELSE 0 END) AS rec,
                     sum(CASE WHEN c.intent='약속' THEN 1 ELSE 0 END) AS appt""", d=d).single()
            ru = s.run("MATCH (m:Memory {day:date($d)}) WHERE m.type='rumor' RETURN count(m) AS c", d=d).single()["c"]
            print(f"  {d}({wlabel(d):>5}) 대화 {c:>5}쌍 · 참여 {p:>4}명({100*p/7500:.0f}%) · "
                  f"추천 {it['rec']}/약속 {it['appt']} · 소문 {ru}")
        # 약속 주입 배선 상태 (should_inject/offset/time 완비 + 대화상대 KNOWS 비율)
        appt = s.run("""MATCH (c:Conversation {intent:'약속'})
          RETURN count(*) AS n,
            sum(CASE WHEN c.should_inject=true AND c.target_day_offset IS NOT NULL
                     AND c.target_time IS NOT NULL THEN 1 ELSE 0 END) AS ready""").single()
        if appt["n"]:
            print(f"  [약속 주입 배선] 약속 {appt['n']}건 中 주입필드 완비 {appt['ready']}건 "
                  f"({100*appt['ready']/appt['n']:.0f}%) — Plan 자동주입 준비율")
        kn = s.run("""MATCH (c:Conversation) WHERE c.day = date($d)
          MATCH (i:Agent {id:c.initiator_id}), (rc:Agent {id:c.recipient_id})
          RETURN count(*) AS tot, sum(CASE WHEN (i)-[:KNOWS]-(rc) THEN 1 ELSE 0 END) AS knows""",
          d=comp[-1] if comp else None).single() if comp else None
        if kn and kn["tot"]:
            print(f"  [대화 상대] {comp[-1]} 기준 KNOWS 관계 {100*kn['knows']/kn['tot']:.0f}% (지인 기반 대화)")


def H_integrity():
    hr("H. Neo4j 무결성")
    from _common import driver_session
    with driver_session() as s:
        for lbl in ["State", "Plan", "Memory", "Conversation"]:
            rows = s.run(f"MATCH (n:{lbl}) WHERE n.day IS NOT NULL "
                         f"RETURN toString(n.day) AS d, count(n) AS c ORDER BY d")
            print(f"  {lbl:13}: " + (", ".join(f"{r['d'][5:]}:{r['c']}" for r in rows) or "없음"))
        for et in ["HAS_STATE", "HAS_PLAN", "INCLUDES", "applied_to"]:
            c = s.run(f"MATCH ()-[r:{et}]->() RETURN count(r) AS c").single()["c"]
            print(f"  edge {et:12}: {c:,}")
        orp = s.run("MATCH (st:State) WHERE NOT (st)<-[:HAS_STATE]-() RETURN count(st) AS c").single()["c"]
        dup = s.run("""MATCH (a:Agent)-[:HAS_STATE]->(st:State)
          WITH a, st.day AS d, count(st) AS c WHERE c > 1 RETURN count(*) AS dup""").single()["dup"]
        print(f"  고아 State: {orp} (0 정상) · State 중복(agent+day): {dup} (0 정상)")


def main():
    days = load_days()
    print(f"■ EXP-001 상세 검증  모델 EXAONE-4.5-33B-AWQ  ({len(days)}일 파일)")
    A_speed(days)
    B_quality(days)
    try:
        C_completeness(days)
        D_consumption(days)
        E_mobility(days)
        F_policy(days)
        G_social(days)
        H_integrity()
    except Exception as e:
        import traceback
        print(f"\n[Neo4j 점검 오류] {e}")
        traceback.print_exc()
    print("\n" + "-" * 72)


if __name__ == "__main__":
    main()
