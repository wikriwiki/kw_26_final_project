"""P009 13일 백업(JSONL) 전체 검증 — 문헌 기반 지표(JSD/cosine/MAPE/Gini/멱법칙) + 정책반응(MPC).

입력: Neo4j 백업 JSONL (includes/state/plan/knows_poi)
정답: BDC temporal_activity_by_demo, dong b069_sales (held-out)
조인: 소득=grant 금액 역산(100k중상/250k중/450k중하/600k하), POI→동=viz 매핑
"""
from __future__ import annotations
import json, io, math, re, zipfile
from collections import defaultdict, Counter
from datetime import date

BK = r"C:/Users/srdyh/Downloads/neo4j_backup_20260607_104307-20260607T035303Z-3-001/neo4j_backup_20260607_104307"
REPO = r"C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a"
OUT = io.open(REPO + "/backup_validation_result_NEW.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

def canon(s): return re.sub(r"제(?=\d)", "", s or "")
def jsonl(path):
    for ln in io.open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try: yield json.loads(ln)
            except: pass

# ---------- helpers ----------
def jsd(P, Q):
    s1 = sum(P) or 1; s2 = sum(Q) or 1
    P = [x/s1 for x in P]; Q = [x/s2 for x in Q]
    M = [(a+b)/2 for a, b in zip(P, Q)]
    kl = lambda A, B: sum(a*math.log2(a/b) for a, b in zip(A, B) if a > 0 and b > 0)
    return 0.5*kl(P, M) + 0.5*kl(Q, M)
def cosine(x, y):
    d = sum(a*b for a, b in zip(x, y)); nx = math.sqrt(sum(a*a for a in x)); ny = math.sqrt(sum(b*b for b in y))
    return d/(nx*ny) if nx*ny else 0
def pear(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y)); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(sx*sy) if sx*sy else 0
def spearman(x, y):
    def rk(v):
        idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for p, i in enumerate(idx): r[i] = p+1
        return r
    return pear(rk(x), rk(y))
def mape(sim, real):
    pairs = [(s, r) for s, r in zip(sim, real) if r > 0]
    return 100*sum(abs(s-r)/r for s, r in pairs)/len(pairs) if pairs else 0
def gini(v):
    v = sorted(v); n = len(v); s = sum(v)
    return (2*sum((i+1)*x for i, x in enumerate(v))/(n*s)) - (n+1)/n if s else 0

INTERNAL = {"집", "직장"}
GRANT2INC = {100000: "중상", 250000: "중", 450000: "중하", 600000: "하"}

# ---------- load plan: pid -> (aid, day) ----------
pid2 = {}
for r in jsonl(BK + "/plan.jsonl"):
    pid2[r["pid"]] = (r["aid"], r["day"])
days = sorted({d for _, d in pid2.values()})
agents = {a for a, _ in pid2.values()}
w("# P009 14일 백업(neo4j_backup_20260607_104307) 검증 결과")
w("기간:", days[0], "~", days[-1], "(%d일)" % len(days), "| agent:", len(agents), "| plan:", len(pid2))

def is_weekend(d):
    y, m, dd = map(int, d.split("-")); return date(y, m, dd).weekday() >= 5

# ---------- state: 소득(grant 역산), grant 잔여 ----------
inc_of = {}; grant_recv = {}; grant_rem_last = {}
for r in jsonl(BK + "/state.jsonl"):
    aid = r["aid"]
    gr = r.get("grant_received") or "{}"
    try: gd = json.loads(gr)
    except: gd = {}
    amt = int(gd.get("P009", 0)) if gd else 0
    if amt > 0:
        inc_of[aid] = GRANT2INC.get(amt, "기타")
        grant_recv[aid] = amt
    rr = r.get("grant_remaining") or "{}"
    try: rem = json.loads(rr)
    except: rem = {}
    if rem.get("P009") is not None:
        grant_rem_last[aid] = int(rem.get("P009", 0))  # 마지막 값으로 덮어씀(날짜순 아님 주의)
w("grant 수령 agent:", len(grant_recv), "| 소득분포:", dict(Counter(inc_of.values())))

# ---------- includes: 시간/소비/카테고리/재방문/정책소비 ----------
hour_all = defaultdict(float); hour_wd = defaultdict(float); hour_we = defaultdict(float)
spend_agent = defaultdict(float); cat_spend = defaultdict(float)
acts_per_plan = Counter(); polspend_agent = defaultdict(float)
n_ev = 0; n_comm = 0
poi_dong_need = set()
ev_for_spatial = []  # (poi_id, spent, aid)
for r in jsonl(BK + "/includes.jsonl"):
    n_ev += 1
    pid = r.get("pid"); aid, day = pid2.get(pid, (None, None))
    cat = r.get("category") or ""
    if cat not in INTERNAL:
        acts_per_plan[pid] += 1
    if cat in INTERNAL or not r.get("poi_id"):
        continue
    n_comm += 1
    tm = r.get("tm") or ""
    sp = r.get("actual_spent") or 0
    if aid:
        spend_agent[aid] += sp
    cat_spend[cat] += sp
    # 시간
    if len(tm) >= 2 and tm[:2].isdigit():
        h = int(tm[:2])
        if 7 <= h <= 21:
            hour_all[h] += 1
            if day:
                (hour_we if is_weekend(day) else hour_wd)[h] += 1
    # 정책소비
    ps = r.get("spent_from_policy") or "{}"
    try: pd = json.loads(ps)
    except: pd = {}
    if pd and aid:
        polspend_agent[aid] += sum(int(v) for v in pd.values())
    # 공간(viz 조인용)
    ev_for_spatial.append((r["poi_id"], sp, aid))
w("이벤트:", n_ev, "| commerce:", n_comm)

# ====== [지표1] 시간 분포 (JSD/cosine/MAPE) ======
W = list(range(7, 22))
G = json.load(io.open(REPO + "/output/stats/global_distributions.json", encoding="utf-8"))
real = defaultdict(float)
for code, hrs in G["temporal_activity_by_demo"].items():
    for hh, val in hrs.items():
        if int(hh) in W: real[int(hh)] += val
sa = [hour_all[h] for h in W]; ra = [real[h] for h in W]
w("\n## [1] 시간대 활동 분포 (정답: BDC temporal_activity_by_demo, held-out)")
w("  cosine=%.3f  JSD=%.4f  MAPE=%.1f%%  피크 시뮬%d시/실측%d시" % (
    cosine([x/sum(sa) for x in sa], [x/sum(ra) for x in ra]), jsd(sa, ra), mape([x/sum(sa) for x in sa], [x/sum(ra) for x in ra]),
    W[sa.index(max(sa))], W[ra.index(max(ra))]))
swd = [hour_wd[h] for h in W]; swe = [hour_we[h] for h in W]
w("  [주말 검증, 신규] 평일↔주말 시간분포 cosine=%.3f JSD=%.4f (시뮬 내부)" % (
    cosine([x/(sum(swd)or 1) for x in swd], [x/(sum(swe)or 1) for x in swe]), jsd(swd, swe)))

# ====== [지표2] 활동사슬 (일 활동수) ======
vals = list(acts_per_plan.values())
w("\n## [2] 활동사슬 — 1일 외출활동 수 (정답: 서울 목적통행 2.59/인·일, 서울연구원 가구통행실태조사. NHTS 폐기)")
w("  평균=%.2f 중앙=%d 분포=%s" % (sum(vals)/len(vals), sorted(vals)[len(vals)//2],
    dict(sorted(Counter(vals).items()))))

# ====== [지표3] 재방문/단골 (knows_poi) ======
vc = Counter()
for r in jsonl(BK + "/knows_poi.jsonl"):
    if (r.get("visit_count") or 0) > 0:
        vc[r["visit_count"]] += 1
tot = sum(vc.values())
w("\n## [3] 재방문·단골 분포 (knows_poi, 정답: 충성도 Zipf 정형사실)")
for k in sorted(vc)[:8]:
    w("  %d회: %.1f%%" % (k, 100*vc[k]/tot))
w("  Gini(방문횟수)=%.3f" % gini([k for k, c in vc.items() for _ in range(c)]))

# ====== [지표4] 소비 불평등 ======
sv = sorted(spend_agent.values())
w("\n## [4] 소비 불평등 (정답: 소득 Gini 0.39 > 소비)")
w("  1인당 소비 Gini=%.3f 상위10%% 점유=%.1f%% (평균 %.0f원)" % (
    gini(sv), 100*sum(sv[int(len(sv)*0.9):])/sum(sv), sum(sv)/len(sv)))

# ====== [지표5] P009 정책반응 (소득별 grant 소비, MPC) — 이 데이터 핵심 ======
w("\n## [5] P009 정책반응 — 소득별 지원금 소비 (MPC) [이 백업의 핵심]")
by_inc = defaultdict(lambda: {"n":0,"grant":0.0,"polspend":0.0,"spend":0.0})
for aid, inc in inc_of.items():
    b = by_inc[inc]; b["n"]+=1; b["grant"]+=grant_recv.get(aid,0)
    b["polspend"]+=polspend_agent.get(aid,0); b["spend"]+=spend_agent.get(aid,0)
w("  %-4s %6s %12s %12s %8s %10s" % ("소득","n","총지원금","정책소비","MPC","1인소비"))
for inc in ["하","중하","중","중상"]:
    if inc in by_inc:
        b=by_inc[inc]; mpc=b["polspend"]/b["grant"] if b["grant"] else 0
        w("  %-4s %6d %12.0f %12.0f %7.1f%% %10.0f" % (inc,b["n"],b["grant"],b["polspend"],100*mpc,b["spend"]/b["n"]))
# 소득-소비 상관 (grant 역산 소득 1~4)
inc_rank={"중상":4,"중":3,"중하":2,"하":1}
xy=[(inc_rank[inc_of[a]], spend_agent.get(a,0)) for a in inc_of if inc_of[a] in inc_rank]
w("  소득-소비 Pearson=%.3f (n=%d, grant 역산 소득)" % (pear([a for a,_ in xy],[b for _,b in xy]), len(xy)))

# ====== [지표6] 공간 분포 (viz POI→동 조인) ======
w("\n## [6] 공간 소비 분포 (정답: dong b069_sales, held-out · viz POI→동 조인)")
z = zipfile.ZipFile(REPO + "/output/sim/visualization/sim_standalone.zip")
h = z.read([n for n in z.namelist() if n.endswith(".html")][0]).decode("utf-8")
def grab(v):
    for ln in h.split("\n"):
        if ln.startswith(v):
            b = ln[len(v):].rstrip(); b = b[:-1] if b.endswith(";") else b; return json.loads(b)
VEV = grab("window.__EVENTS__ = ")
poi2dong = {}
for evs in VEV.values():
    for e in evs:
        if e.get("poi_id") and e.get("dong"): poi2dong[e["poi_id"]] = e["dong"]
w("  viz POI→동 매핑:", len(poi2dong), "개")
# 동명→코드
P = json.load(io.open(REPO + "/output/stats/agent_profiles.json", encoding="utf-8"))
n2c = {}; pop_gu = defaultdict(float)
for v in P.values():
    loc = v.get("location", {})
    if loc.get("dong") and loc.get("adm_cd_8"): n2c[canon(loc["dong"])] = loc["adm_cd_8"]
    if loc.get("adm_cd_8"): pop_gu[loc["adm_cd_8"][:5]] += v.get("demographics", {}).get("population", 0)
# 백업 agent gu 분포 (균등 여부)
agu = Counter(a.split("_")[1][:5] for a in agents)
w("  백업 agent gu 분포: min=%d max=%d (균등이면 인구가중 필요)" % (min(agu.values()), max(agu.values())))
nsamp = agu
M = defaultdict(float); matched = 0; unm = 0
for poi, sp, aid in ev_for_spatial:
    dn = poi2dong.get(poi)
    if not dn: unm += 1; continue
    cd = n2c.get(canon(dn))
    if not cd: unm += 1; continue
    matched += 1
    rg = aid.split("_")[1][:5] if aid else None
    wgt = pop_gu.get(rg, 0)/max(nsamp.get(rg, 1), 1)
    M[cd[:5]] += sp*wgt
w("  공간 조인 커버리지: %d/%d (%.1f%%)" % (matched, matched+unm, 100*matched/(matched+unm) if matched+unm else 0))
DC = json.load(io.open(REPO + "/output/stats/dong_context.json", encoding="utf-8"))
O = defaultdict(float)
for k, v in DC.items():
    if isinstance(v, dict) and v.get("b069_sales") is not None: O[k[:5]] += v["b069_sales"]
gus = sorted(set(M) & set(O)); mv = [M[g] for g in gus]; ov = [O[g] for g in gus]
if mv and sum(mv) > 0:
    nm = [x/sum(mv) for x in mv]; no = [x/sum(ov) for x in ov]
    w("  [공간] 자치구 %d개: cosine=%.3f JSD=%.4f MAPE=%.1f%% Pearson=%.3f Spearman=%.3f" % (
        len(gus), cosine(nm, no), jsd(mv, ov), mape(nm, no), pear(mv, ov), spearman(mv, ov)))
    w("  [공간집중 Gini] 시뮬=%.3f 실측=%.3f" % (gini(mv), gini(ov)))
else:
    w("  [공간] 조인 실패 — 매핑 커버리지 부족")

OUT.close()
print("DONE")
