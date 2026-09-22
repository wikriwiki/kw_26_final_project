"""한국 데이터 기반 재검증 (Task A: NHTS 폐기 → KOTI/BDC 정답)
- 외출/통행 횟수: 시뮬을 KOTI '통행(장소 이동)' 정의로 재계산 → 서울 통행원단위(서울연구원 2022)와 비교
- 소비 지니: 시뮬 vs BDC weekday_spending_level decile 실측
- 공간 지니: 시뮬 vs BDC dong b069_sales 실측 (동/자치구)
입력: 새 14일 백업 JSONL + output/stats(BDC 가공) + viz POI→동 매핑
"""
from __future__ import annotations
import json, io, math, re, zipfile
from collections import defaultdict, Counter

BK   = r"C:/Users/srdyh/Downloads/neo4j_backup_20260607_104307-20260607T035303Z-3-001/neo4j_backup_20260607_104307"
REPO = r"C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a"
OUT  = io.open(REPO + "/korea_validation_result.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

def canon(s): return re.sub(r"제(?=\d)", "", s or "")
def jsonl(path):
    for ln in io.open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try: yield json.loads(ln)
            except: pass
def gini(v):
    v = sorted(x for x in v if x is not None); n = len(v); s = sum(v)
    return (2*sum((i+1)*x for i, x in enumerate(v))/(n*s)) - (n+1)/n if s and n else 0
def cosine(x, y):
    d = sum(a*b for a, b in zip(x, y)); nx = math.sqrt(sum(a*a for a in x)); ny = math.sqrt(sum(b*b for b in y))
    return d/(nx*ny) if nx*ny else 0
def jsd(P, Q):
    s1 = sum(P) or 1; s2 = sum(Q) or 1
    P=[x/s1 for x in P]; Q=[x/s2 for x in Q]; M=[(a+b)/2 for a,b in zip(P,Q)]
    kl=lambda A,B: sum(a*math.log2(a/b) for a,b in zip(A,B) if a>0 and b>0)
    return 0.5*kl(P,M)+0.5*kl(Q,M)

INTERNAL = {"집", "직장"}

# ---------- plan: pid -> (aid, day) ----------
pid2 = {}
for r in jsonl(BK + "/plan.jsonl"):
    pid2[r["pid"]] = (r["aid"], r["day"])
agents = set(a for a,_ in pid2.values())
days = sorted(set(d for _,d in pid2.values()))
w("# 한국 데이터 기반 재검증")
w("기간:", days[0], "~", days[-1], "(%d일)" % len(days), "| agent:", len(agents), "| plan:", len(pid2))

# ---------- state: 소득(grant 직접) ----------
GRANT2INC = {100000:"중상", 250000:"중", 450000:"중하", 600000:"하"}
inc_of = {}
for r in jsonl(BK + "/state.jsonl"):
    try: gr = json.loads(r.get("grant_received") or "{}")
    except: gr = {}
    amt = gr.get("P009")
    if amt: inc_of[r["aid"]] = GRANT2INC.get(int(amt), "?")
w("grant 수령 agent:", len(inc_of), "| 소득분포:", dict(Counter(inc_of.values())))

# ---------- includes: 통행/소비/공간 ----------
plan_rows = defaultdict(list)            # pid -> [(ord, poi_id, category)]
spend_agent = defaultdict(float)         # aid -> total spent
ev_for_spatial = []                      # (poi_id, spent, aid)
n_ev = 0
for r in jsonl(BK + "/includes.jsonl"):
    n_ev += 1
    pid = r.get("pid")
    plan_rows[pid].append((r.get("ord", 0), r.get("poi_id"), r.get("category") or ""))
    sp = r.get("actual_spent") or 0
    aid, day = pid2.get(pid, (None, None))
    if (r.get("category") or "") not in INTERNAL and r.get("poi_id"):
        if aid: spend_agent[aid] += sp
        ev_for_spatial.append((r["poi_id"], sp, aid))
w("includes 이벤트:", n_ev, "| plan(일):", len(plan_rows))

# ====== [A] 통행(이동) 횟수 — KOTI 정의 ======
trips = []        # 통행수 = 장소 이동 횟수
visits = []       # 비거주/직장 방문지(고유 POI) 수
old_act = []      # 기존 '외출 활동 수'(집/직장 제외 행 수)
for pid, rows in plan_rows.items():
    rows.sort(key=lambda x: x[0])
    seq = [pid_poi for _, pid_poi, _ in rows]
    # 통행: 연속 행에서 위치(poi)가 바뀐 횟수
    t = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1] and seq[i] and seq[i-1])
    trips.append(t)
    visits.append(len(set(p for _, p, c in rows if c not in INTERNAL and p)))
    old_act.append(sum(1 for _, p, c in rows if c not in INTERNAL))

def stat(name, arr):
    arr2 = sorted(arr); n=len(arr2)
    w("  %s: 평균=%.2f 중앙=%d 분포=%s" % (
        name, sum(arr2)/n, arr2[n//2], dict(sorted(Counter(arr2).items()))))

w("\n## [A] 1일 통행/외출 횟수 (정답: 서울 통행원단위 ≈ 3.7 통행/인·일, 서울연구원 2022)")
stat("통행수(장소 이동 횟수, KOTI 정의)", trips)
stat("비거주 방문지 수(고유 POI)", visits)
stat("기존 외출활동수(집/직장 제외 행)", old_act)
# 평일/주말 분리(통행수)
def is_we(d):
    from datetime import date
    y,m,dd=map(int,d.split("-")); return date(y,m,dd).weekday()>=5
twd=[]; twe=[]
for pid, rows in plan_rows.items():
    _, day = pid2.get(pid,(None,None))
    rows.sort(key=lambda x:x[0]); seq=[p for _,p,_ in rows]
    t=sum(1 for i in range(1,len(seq)) if seq[i]!=seq[i-1] and seq[i] and seq[i-1])
    (twe if (day and is_we(day)) else twd).append(t)
w("  평일 통행수 평균=%.2f / 주말 통행수 평균=%.2f" % (sum(twd)/len(twd), sum(twe)/len(twe)))

# ====== [B] 소비 지니: 시뮬 vs BDC 실측 ======
w("\n## [B] 소비 불평등 지니 — 시뮬 vs BDC 실측")
sv = list(spend_agent.values())
w("  [시뮬] 1인당 누적소비 Gini=%.3f (n=%d, 평균 %.0f원)" % (gini(sv), len(sv), sum(sv)/len(sv)))
# 1인·1일 환산
per_day = [s/len(days) for s in sv]
w("  [시뮬] 1인·1일 소비 Gini=%.3f (평균 %.0f원/일)" % (gini(per_day), sum(per_day)/len(per_day)))
# BDC 실측: weekday_spending_level decile 중간값 (10등분 = 인구 10%씩)
db = json.load(io.open(REPO + "/output/stats/decile_boundaries.json", encoding="utf-8"))
for key,label in [("weekday_spending_level","평일"),("weekend_spending_level","주말")]:
    mids = [ (b["min"]+b["max"])/2 for b in db[key]["boundaries"] ]
    # 10개 동일 인구집단 → 각 집단을 동일 가중으로 Gini
    w("  [BDC실측] %s 1인·1일 소비 Gini=%.3f (decile 중간값, 단위 원: %s)" % (
        label, gini(mids), [round(m) for m in mids]))

# ====== [C] 공간 지니: 시뮬 vs BDC 실측 ======
w("\n## [C] 공간(소비 집중) 지니 — 시뮬 vs BDC 실측")
DC = json.load(io.open(REPO + "/output/stats/dong_context.json", encoding="utf-8"))
dong_sales = {k: v["b069_sales"] for k,v in DC.items() if isinstance(v,dict) and v.get("b069_sales") is not None}
gu_sales = defaultdict(float)
for k,s in dong_sales.items(): gu_sales[k[:5]] += s
w("  [BDC실측] 동별 매출(b069_sales) Gini=%.3f (n=%d개 동)" % (gini(list(dong_sales.values())), len(dong_sales)))
w("  [BDC실측] 자치구별 매출 Gini=%.3f (n=%d개 구)" % (gini(list(gu_sales.values())), len(gu_sales)))

# 시뮬 공간: viz POI→동 매핑 + 인구가중
try:
    z = zipfile.ZipFile(REPO + "/output/sim/visualization/sim_standalone.zip")
    h = z.read([n for n in z.namelist() if n.endswith(".html")][0]).decode("utf-8")
except Exception:
    h = io.open(REPO + "/output/sim/visualization/sim_standalone.html", encoding="utf-8").read()
def grab(v):
    for ln in h.split("\n"):
        if ln.startswith(v):
            b = ln[len(v):].rstrip(); b = b[:-1] if b.endswith(";") else b; return json.loads(b)
VEV = grab("window.__EVENTS__ = ") or {}
poi2dong = {}
for evs in VEV.values():
    for e in evs:
        if e.get("poi_id") and e.get("dong"): poi2dong[e["poi_id"]] = e["dong"]
P = json.load(io.open(REPO + "/output/stats/agent_profiles.json", encoding="utf-8"))
n2c = {}; pop_gu = defaultdict(float)
for v in P.values():
    loc = v.get("location", {})
    if loc.get("dong") and loc.get("adm_cd_8"): n2c[canon(loc["dong"])] = loc["adm_cd_8"]
    if loc.get("adm_cd_8"): pop_gu[loc["adm_cd_8"][:5]] += v.get("demographics", {}).get("population", 0)
nsamp = Counter(a.split("_")[1][:5] for a in agents)
Msim_gu = defaultdict(float); Msim_dong = defaultdict(float); matched=0; unm=0
for poi, sp, aid in ev_for_spatial:
    dn = poi2dong.get(poi)
    if not dn: unm+=1; continue
    cd = n2c.get(canon(dn))
    if not cd: unm+=1; continue
    matched+=1
    rg = aid.split("_")[1][:5] if aid else None
    wgt = pop_gu.get(rg,0)/max(nsamp.get(rg,1),1)
    Msim_gu[cd[:5]] += sp*wgt
    Msim_dong[cd] += sp*wgt
w("  공간 조인 커버리지: %d/%d (%.1f%%)" % (matched, matched+unm, 100*matched/(matched+unm) if matched+unm else 0))
w("  [시뮬] 자치구별 소비 Gini=%.3f (n=%d)" % (gini(list(Msim_gu.values())), len(Msim_gu)))
w("  [시뮬] 동별 소비 Gini=%.3f (n=%d)" % (gini(list(Msim_dong.values())), len(Msim_dong)))
# 동일 자치구 집합에서 cosine 비교
gus = sorted(set(Msim_gu) & set(gu_sales))
if gus:
    mv=[Msim_gu[g] for g in gus]; ov=[gu_sales[g] for g in gus]
    w("  [공간 형태일치] 자치구 %d개 cosine=%.3f JSD=%.4f" % (len(gus), cosine(mv,ov), jsd(mv,ov)))

OUT.close(); print("DONE")
