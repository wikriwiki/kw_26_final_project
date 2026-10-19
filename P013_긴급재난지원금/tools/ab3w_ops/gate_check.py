# 본런 진입 검수 (읽기 전용). python3 gate_check.py <run> <on_bolt> <off_bolt> <policy:p012|p013> <first_policy_day>
import collections, glob, json, math, os, re, statistics as st, sys
from neo4j import GraphDatabase

run, bon, boff, pol, d0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
B = f"/data/ab3w/{run}"
PW = os.environ["NEO4J_PASSWORD_ON"]
res = {}


def mark(k, ok, msg):
    res[k] = (ok, msg)
    print(f"[{'통과' if ok else '실패' if ok is False else '참고'}] {k}: {msg}")


def load_metrics(arm):
    out = {}
    for f in sorted(glob.glob(f"{B}/{arm}/metrics/day_*.jsonl")):
        d = os.path.basename(f)[4:14]
        out[d] = [json.loads(l) for l in open(f)]
    return out


M = {a: load_metrics(a) for a in ("pre", "on", "off")}
Q = """MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[r:INCLUDES]->(x:POI)
       RETURN a.id AS aid, toString(p.day) AS d, r.category AS cat, r.sub_category AS sub, r.menu AS menu,
              r.unit_price AS up, r.pay_count AS pc, r.actual_spent AS amt, r.budget_skipped AS skip,
              r.purchase_status AS ps, r.reasoning AS why, r.actual_satisfaction AS sat,
              toString(r.time) AS t, r.trigger AS trig, r.intent AS intent, x.name AS poi, x.id AS poi_id,
              head([(x)-[:IN_CATEGORY]->(oc:Category) | oc.name]) AS poi_sub"""
QM = "MATCH (a:Agent)-[:REMEMBERS]->(m:Memory) WHERE m.type='visited' RETURN a.id AS aid, toString(m.day) AS d, m.summary AS s, m.menu AS menu"
QS = "MATCH (a:Agent)-[:HAS_STATE]->(s:State) RETURN a.id AS aid, toString(s.day) AS d, s.fatigue AS f, s.balance AS bal, s.budget_carry AS carry"
QA = """MATCH (a:Agent) WHERE a.id IN $ids RETURN a.id AS aid, a.p_income_level AS inc, a.p_age_group AS age,
        a.p_life_stage AS ls, a.cat_ratio_wd AS crw, a.cat_ratio_we AS cre"""


def pull(bolt, ids):
    drv = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", PW))
    with drv.session() as s:
        r = (s.run(Q).data(), s.run(QM).data(), s.run(QS).data(), s.run(QA, ids=ids).data())
    drv.close()
    return r


ids = sorted({m["aid"] for ms in M["pre"].values() for m in ms})
on_r, on_m, on_s, agents = pull(bon, ids)
_mine = set(M["pre"]) | set(M["on"])
on_r = [r for r in on_r if r["d"] in _mine]
on_s = [x for x in on_s if x["d"] in _mine]
forked = os.path.exists(f"{B}/fork.marker")
# 복제 전의 정책 없음 그래프에는 다른 런의 옛 데이터가 남아 있다 — 복제 뒤에만 읽고, 이 런의 날짜만 쓴다.
if forked:
    off_r, off_m, off_s, _ = pull(boff, ids)
    off_r = [r for r in off_r if r["d"] in M["off"]]
    off_s = [x for x in off_s if x["d"] in M["off"]]
else:
    off_r, off_m, off_s = [], [], []

# ---------------- A
allm = [(a, d, m) for a in M for d, ms in M[a].items() for m in ms]
bad = [m for _, _, m in allm if m.get("status") != "ok"]
n_days = {a: len(M[a]) for a in M}
mark("1 사람-날 처리", not bad and all(len(ms) == len(ids) for a in M for ms in M[a].values()),
     f"사람 {len(ids)} · 날 {n_days} · status≠ok {len(bad)}")

acc_bad = 0
for _, _, m in allm:
    for e in m.get("execution_receipts") or []:
        if e.get("kind") != "purchase_receipt":
            continue
        a, o = e.get("amount") or 0, e.get("own_paid") or 0
        if a < 0 or o < 0 or o > a:
            acc_bad += 1
# 잔액식: 다음날 잔액 = 잔액 + 소득 - 본인결제 - 온라인 (같은 갈래 연속일)
ident_bad = ident_n = 0
for arm in ("pre", "on", "off"):
    days = sorted(M[arm])
    for d1, d2 in zip(days, days[1:]):
        a1 = {m["aid"]: m for m in M[arm][d1]}
        for m2 in M[arm][d2]:
            m1 = a1.get(m2["aid"])
            if not m1 or m1.get("balance") is None or m2.get("balance") is None:
                continue
            own = sum(e.get("own_paid") or 0 for e in m1["execution_receipts"] if e.get("kind") == "purchase_receipt")
            own = sum(e.get("own_paid") or 0 for e in m2["execution_receipts"] if e.get("kind") == "purchase_receipt")
            exp = max(0, m1["balance"] + (m2.get("cm_income_today") or 0) - own - (m2.get("cm_online_total") or 0))  # 지표 잔액 = 그날 끝 값
            ident_n += 1
            ident_bad += abs(m2["balance"] - exp) > 2
mark("2 회계", acc_bad == 0 and ident_bad == 0, f"음수·초과 결제 {acc_bad} · 잔액식 불일치 {ident_bad}/{ident_n}")

if pol == "p013":
    g = sum(m.get("grant_applied_today") or 0 for ms in M["on"].values() for m in ms)
    used = sum(m.get("policy_spend_today") or 0 for ms in M["on"].values() for m in ms)
    last = max(M["on"]) if M["on"] else None
    rem = sum(m.get("grant_remaining_total") or 0 for m in M["on"].get(last, []))
    mark("3 정책 정산", g - used == rem and g > 0, f"지급 {g:,} − 사용 {used:,} = {g-used:,} / 잔여 {rem:,}")
elif pol == "distancing":
    # [2026-10-08] 환경형: 정책 돈이 없다. 두 갈래 거리두기 원장이 사람×날을 다 덮는지 본다.
    _rows = {a: sum(1 for _ in open(f"{B}/{a}/distancing.ledger.jsonl")) if os.path.exists(f"{B}/{a}/distancing.ledger.jsonl") else 0 for a in ("on", "off")}
    _need = len(set(m["aid"] for ms in M["on"].values() for m in ms)) * len(M["on"])
    mark("3 정책 정산", _rows["on"] == _need and _rows["off"] == _need and _need > 0, f"거리두기 원장 행 {_rows} / 필요 {_need}(환경형, 정책 돈 없음)")
else:
    led = f"{B}/on/cashback.ledger.jsonl"
    ok = os.path.exists(led) and os.path.getsize(led) > 0
    rows = [json.loads(l) for l in open(led)] if ok else []
    keys = list(rows[0].keys())[:14] if rows else []
    mark("3 정책 정산", ok and len(rows) > 0, f"캐시백 원장 {len(rows)}행 · 칸 {keys}")

off_pol = sum(1 for ms in M["off"].values() for m in ms
              if (m.get("policy_spend_today") or 0) or (m.get("grant_applied_today") or 0) or (m.get("policy_rebate_today") or 0)
              or any(e.get("policy_facts") for e in m["execution_receipts"]))
mark("4 정책 없는 쪽", off_pol == 0, f"정책 돈·기록 {off_pol}")

need = ["on/sector.ledger.jsonl", "off/sector.ledger.jsonl", "on/dossier.jsonl", "off/dossier.jsonl"]
need += (["on/policy.ledger.jsonl", "off/policy.ledger.jsonl"] if pol == "p013" else ["on/distancing.ledger.jsonl", "off/distancing.ledger.jsonl"] if pol == "distancing" else ["on/cashback.ledger.jsonl", "off/cashback.ledger.jsonl"])
missing = [p for p in need if not (os.path.exists(f"{B}/{p}") and os.path.getsize(f"{B}/{p}") > 0)]
score = glob.glob(f"{B}/score/*.json")
endok = "=== 끝" in open(f"{B}/orchestrate.log").read()
mark("5 결과 내보내기", not missing and bool(score) and endok, f"빠진 파일 {missing} · 채점 {len(score)}개 · 끝 표지 {endok}")

# 기억: 결제(만족도 있는) 방문 수 = visited 기억 수 (on 그래프 기준, 정책 전+있음)
paid_on = [r for r in on_r if r["cat"] not in ("집", "직장") and r["sat"] is not None]
mem_on = len(on_m)
menu_mem = sum(1 for m in on_m if m["menu"])
mark("6 기억·계획 저장", mem_on == len(paid_on) and menu_mem > 0,
     f"만족도 있는 외출 {len(paid_on)} · 방문 기억 {mem_on} · 메뉴 담긴 기억 {menu_mem}")

# ---------------- B
rows = on_r + [r for r in off_r if r["d"] >= d0]
out = [r for r in rows if r["cat"] not in ("집", "직장")]
paid = [r for r in out if (r["amt"] or 0) > 0]
by = collections.defaultdict(list)
for r in paid:
    by[r["sub"]].append(r["amt"])
solo_meal = [r["amt"] for r in paid if r["cat"] == "식사" and (r["pc"] or 1) == 1]
pc_max = max([r["pc"] or 1 for r in paid] or [0])
txt = ", ".join(f"{k} {st.median(v):,.0f}" for k, v in sorted(by.items(), key=lambda x: -len(x[1]))[:8])
ok7 = bool(solo_meal) and 5000 <= st.median(solo_meal) <= 20000 and pc_max <= 8
mark("7 건당 금액", ok7, f"혼자 식사 중앙 {st.median(solo_meal) if solo_meal else 0:,.0f} · 인원 최대 {pc_max} · 세부업종 중앙 {txt}")

ess_skip = sum(1 for r in out if r["skip"] and r["cat"] in ("식사", "마트", "편의점"))
mark("8 고정 생활비", ess_skip == 0, f"끼니·장보기 예산으로 거름 {ess_skip}")

# 예외 지출: 같은 사람, 병원 간 날 vs 안 간 날 하루 가게 결제
day_tot = collections.defaultdict(float)
clinic_day = set()
for r in paid:
    day_tot[(r["aid"], r["d"])] += r["amt"]
    if r["sub"] in ("의원", "치과", "한의원"):
        clinic_day.add((r["aid"], r["d"]))
diffs = []
for a in ids:
    on_d = [day_tot[k] for k in day_tot if k[0] == a and k in clinic_day]
    off_d = [day_tot[k] for k in day_tot if k[0] == a and k not in clinic_day]
    if on_d and off_d:
        diffs.append(st.mean(on_d) - st.mean(off_d))
mark("9 예외 지출이 총액을 올림", (not diffs) or st.mean(diffs) > 0,
     f"병원 간 날 − 안 간 날(같은 사람 {len(diffs)}명) {st.mean(diffs) if diffs else 0:+,.0f}원" + (" (해당자 없음)" if not diffs else ""))

monthly = collections.Counter((r["aid"], r["d"][:7], r["sub"]) for r in paid if r["sub"] in ("학원", "기타교육", "헬스장"))
dup = {k: v for k, v in monthly.items() if v > 1}
att0 = sum(1 for r in out if r["sub"] in ("학원", "기타교육", "헬스장") and not (r["amt"] or 0))
mark("10 월 납부", not dup, f"달에 두 번 이상 낸 경우 {len(dup)} · 낸 건 {sum(monthly.values())} · 0원 이용 {att0}")

clin = collections.defaultdict(set)
for (a, d) in clinic_day:
    clin[a].add(d)
consec = sum(1 for a, ds in clin.items() for d in ds if any(abs((__import__('datetime').date.fromisoformat(d) - __import__('datetime').date.fromisoformat(e)).days) == 1 for e in ds))
pdays = len({(r["aid"], r["d"]) for r in rows})
rate = len(clinic_day) / max(1, pdays)
mark("11 의원 빈도", consec == 0 and rate <= 0.12, f"의원 간 사람-날 {len(clinic_day)}/{pdays} = {100*rate:.1f}% · 이틀 연속 {consec}")

fat = [x["f"] for x in on_s + off_s if x["f"] is not None]
mark("12 피로", bool(fat) and max(fat) < 0.95 and st.mean(fat) < 0.7, f"평균 {st.mean(fat):.2f} · 최대 {max(fat):.2f}")

whys = [r["why"] or "" for r in rows]
cp = {"허리 통증": sum("허리" in w for w in whys), "냉장고 사흘째": sum("사흘째" in w and "냉장고" in w for w in whys),
      "전기밥솥": sum("밥솥" in w for w in whys), "아이 운동화": sum("운동화" in w for w in whys)}
mark("13 예시 문장 반복", sum(cp.values()) <= max(2, len(whys) // 200), f"{cp} / 이유 {len(whys)}개")

# ---------------- C
pre_days = [d for d in M["on"] if True]
pairs = []
for d in sorted(set(M["on"]) & set(M["off"])):
    a1 = {m["aid"]: m for m in M["on"][d]}
    for m2 in M["off"][d]:
        m1 = a1.get(m2["aid"])
        if m1 and not (m1.get("grant_remaining_total") or m1.get("grant_applied_today")) and pol == "p013":
            t = lambda m: (m.get("cm_today_total") or 0) + (m.get("cm_online_total") or 0)
            pairs.append((t(m1), t(m2)))
if pairs:
    sd = st.pstdev([a - b for a, b in pairs]); lvl = st.mean([x for p in pairs for x in p])
    se = sd / math.sqrt(2000 * 7)
    mark("15 검출력", 1.96 * se / lvl < 0.03, f"조건 같은 쌍 {len(pairs)} · 2,000명×7일 95% 범위 ±{196*se/lvl:.2f}%")
else:
    mark("15 검출력", None, "조건 같은 쌍 없음(캐시백은 첫날부터 정책 상태가 다름) — P013 추정 참고")

log = open(f"{B}/orchestrate.log").read()
m = re.findall(r"계획 기준선이 쓰인 것 (\d+) \(([\d.]+)%\)", log)
mark("16 정책 반응 통로", bool(m) and all(float(p) >= 99 for _, p in m), f"기준선 사용 {m}")

tot = [(m.get("cm_today_total") or 0) + (m.get("cm_online_total") or 0) for _, _, m in allm]
mark("17 1인 하루 총액", 25000 <= st.mean(tot) <= 110000, f"평균 {st.mean(tot):,.0f} · 중앙 {st.median(tot):,.0f} (가게+방문 외)")

# ---------------- D (2026-10-07b 추가: 지어낸 값·예시 쏠림·공휴일·약속·산책)
outs = [r for r in rows if r["cat"] not in ("집", "직장")]
paid2 = [r for r in outs if (r["amt"] or 0) > 0]
no_src = [r for r in paid2 if r["up"] is None or r["pc"] is None or abs((r["up"] or 0) * (r["pc"] or 0) - r["amt"]) > 1]
mark("18 금액 출처 = 모델 답", not no_src,
     f"결제 {len(paid2)} 중 단가×인원과 다른 금액 {len(no_src)} (예: {[(r['sub'], r['menu'], r['up'], r['pc'], r['amt']) for r in no_src[:3]]})")
zero_over = [r for r in outs if r["up"] == 0 and (r["amt"] or 0) > 0]
mark("19 0원 존중", not zero_over, f"모델 0원인데 돈 낸 외출 {len(zero_over)} · 0원 외출(거름 제외) {sum(1 for r in outs if r['up'] == 0)}")
first = {}
for r in rows:
    k = (r["aid"], r["d"])
    if r["t"] and (k not in first or r["t"] < first[k]):
        first[k] = r["t"]
wake = collections.Counter(v[:5] for v in first.values())
meals = collections.Counter(r["sub"] for r in paid2 if r["cat"] == "식사")
w_top = wake.most_common(1)[0] if wake else ("", 0)
mark("20 예시 값 쏠림(참고)", None,
     f"첫 일정 시각 상위 {wake.most_common(4)} (최다 {100*w_top[1]/max(1,len(first)):.0f}%) · 외식 세부업종 {meals.most_common(4)}")
from datetime import date as _date
sys.path.insert(0, "/data/repo_ab3w_20261007c/scripts/sim")
from kr_holidays import is_day_off as _off
work_days = collections.defaultdict(set)
for r in rows:
    if r["cat"] == "직장":
        work_days[r["aid"]].add(r["d"])
workers = {a for a, ds in work_days.items() if any(not _off(_date.fromisoformat(d)) for d in ds)}
off_days = sorted({r["d"] for r in rows if _off(_date.fromisoformat(r["d"]))})
off_work = [(a, d) for d in off_days for a in workers if d in work_days[a]]
off_base = len(workers) * len(off_days)
mark("21 쉬는 날 출근(직장인)", (not off_base) or len(off_work) / off_base <= 0.15,
     f"쉬는 날 {off_days} 직장인 {len(workers)}명 중 출근 사람-날 {len(off_work)}/{off_base}")
walk_bad = [r for r in outs if re.search(r"산책|걷기|공원", (r["intent"] or "")) and (r["poi_sub"] or r["sub"]) in ("스포츠", "여행사", "유원지·오락", "노래방", "당구")]
walk_all = [r for r in outs if re.search(r"산책|걷기|공원", (r["intent"] or ""))]
mark("22 산책이 엉뚱한 가게", len(walk_bad) <= max(2, 0.05 * len(walk_all)), f"산책 의도 {len(walk_all)} 중 스포츠·여행사·오락 가게 {len(walk_bad)}")
odd = [r for r in rows if re.search(r"AGT_|\bmood\b|\bsat\s*0\.|fatigue", r["why"] or "")]
mark("23 이유 속 내부 표기", len(odd) <= 0.05 * max(1, len(rows)), f"agent_id·mood·sat·fatigue 표기 이유 {len(odd)}/{len(rows)}")
trig_empty = sum(1 for r in outs if not r["trig"])
mark("24 계기 빈값(참고)", None, f"외출 계기 빈값 {trig_empty}/{len(outs)} · 계기 분포 {collections.Counter(r['trig'] for r in outs).most_common(6)}")

# ---------------- 페르소나별
A = {a["aid"]: a for a in agents}
print("\n== 페르소나별 (정책 전 + 정책 없음 쪽, 사람-날 평균)")
base = [(a, d, m) for a, d, m in allm if a in ("pre", "off")]
for key, name in (("inc", "소득"), ("age", "연령"), ("ls", "생애주기")):
    g = collections.defaultdict(list)
    for _, _, m in base:
        g[A.get(m["aid"], {}).get(key)].append((m.get("cm_today_total") or 0) + (m.get("cm_online_total") or 0))
    print(f"  {name}: " + " · ".join(f"{k} {st.mean(v):,.0f}(n{len(v)})" for k, v in sorted(g.items(), key=lambda x: str(x[0]))))
# BDC 평소 업종 1위가 실제 결제 비중에서도 높은지
hit = tot_p = 0
L1 = {"한식": "식사", "기타요식": "식사", "양식": "식사", "중식": "식사", "일식": "식사", "편의점": "편의점", "커피전문점": "카페",
      "할인점/슈퍼마켓": "마트", "슈퍼마켓": "마트", "학원": "교육", "제과점": "디저트"}
for a in ids:
    try:
        cr = json.loads(A[a]["crw"] or "{}")
    except Exception:
        continue
    top = [L1.get(k) for k, _ in sorted(cr.items(), key=lambda x: -x[1]) if L1.get(k)]
    if not top:
        continue
    spend = collections.Counter()
    for r in paid:
        if r["aid"] == a:
            spend[r["cat"]] += r["amt"]
    if spend:
        tot_p += 1
        hit += spend.most_common(1)[0][0] in top[:2]
print(f"  BDC 평소 업종 상위 2개 안에 실제 최다 지출 업종이 든 사람 {hit}/{tot_p}")
cats = collections.defaultdict(collections.Counter)
for r in paid:
    cats[A.get(r["aid"], {}).get("age")][r["cat"]] += r["amt"]
for k, c in sorted(cats.items(), key=lambda x: str(x[0])):
    t = sum(c.values()) or 1
    print(f"  연령 {k}: " + ", ".join(f"{x} {100*v/t:.0f}%" for x, v in c.most_common(4)))
fails = [k for k, (ok, _) in res.items() if ok is False]
print("\n판정:", "전부 통과" if not fails else f"실패 {fails}")
