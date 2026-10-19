#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시뮬 정밀 점검 — 항목별 이상징후를 하나씩 판정한다.

사용:  python3 health_check.py <RUN>        (기본 BASE7500)

판정: [OK] 정상 · [주의] 추세를 봐야 함 · [이상] 조치 필요 · [정보] 판정 없이 값만
"""
import sys, os, json, glob, subprocess, statistics as st
from collections import defaultdict, Counter

sys.path.insert(0, '/data/exp001_repo/scripts/neo4j_load')
from _common import driver_session   # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'BASE7500'
OUT = f'/data/exp001/out_{RUN}'
LOG = f'/data/exp001/run_{RUN}.log'
N_TARGET = 7500

T = {'OK': 0, '주의': 0, '이상': 0}


def mark(tag, name, detail):
    if tag in T:
        T[tag] += 1
    print(f"  [{tag}] {name:<26} {detail}")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


print(f"\n{'='*78}\n  {RUN} 정밀 점검\n{'='*78}")

# ══ A. 프로세스·인프라 ═══════════════════════════════════════════════════
print("\n── A. 프로세스 · 인프라 ──")
alive = sh("pgrep -f 'run_simulation[.]py' | head -1")
mark('OK' if alive else '이상', 'A1 시뮬 프로세스', f"PID {alive}" if alive else "죽음")
for nm, pat in [('A2 체인', 'chain_p[0-9a-z]*[.]sh'), ('A3 원장감시', 'watch_export'),
                ('A4 SGLang', 'launch_server')]:
    p = sh(f"pgrep -f '{pat}' | head -1")
    mark('OK' if p else '이상', nm, f"PID {p}" if p else "죽음")

gpu = sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
         "--format=csv,noheader").splitlines()
for g in gpu:
    idx, util, used, tot = [x.strip() for x in g.split(',')]
    u = int(util.rstrip(' %'))
    mark('OK' if u >= 60 else '주의', f'A6 GPU{idx} 사용률', f"{util} · {used}/{tot}")

# Neo4j 는 프로세스 매칭이 경로 문자열에 오탐한다(실제로 죽었는데 살아있다고 나옴).
# bolt 포트 응답으로 판정한다.
_nb = sh("/data/neo4j-community-5.26.0/bin/cypher-shell -a bolt://localhost:7687 "
         "-u neo4j -p exp001pass 'RETURN 1' 2>&1 | head -1")
mark('OK' if '1' in _nb else '이상', 'A5 Neo4j(bolt)', '응답 정상' if '1' in _nb else _nb[:60])

df = sh("df -h /data | tail -1").split()
mark('OK' if df[4].rstrip('%').isdigit() and int(df[4].rstrip('%')) < 85 else '주의',
     'A7 디스크', f"{df[2]} 사용 / {df[3]} 여유 ({df[4]})")

q = sh("grep -oE '#running-req: [0-9]+, #queue-req: [0-9]+' /data/sglang.log | tail -1")
qn = int(q.split('#queue-req: ')[1]) if '#queue-req: ' in q else -1
mark('OK' if 0 <= qn <= 60 else ('주의' if qn > 60 else '이상'), 'A8 서버 대기열', q or '읽기 실패')
kv = sh("grep -oE 'token usage: [0-9.]+' /data/sglang.log | tail -1")
mark('정보', 'A9 KV 사용률', kv or '-')

# ══ B. 처리 건전성 ═══════════════════════════════════════════════════════
print("\n── B. 처리 건전성 ──")
fatal = int(sh(f"grep -ciE 'traceback|Killed|OutOfMemory|CUDA error' {LOG}") or 0)
mark('OK' if fatal == 0 else '이상', 'B1 치명 오류', f"{fatal}건")

errs = sh(f"grep -oE 'err=[0-9]+' {LOG} | tail -1")
en = int(errs.split('=')[1]) if '=' in errs else 0
prog = sh(f"grep -oE '[0-9]+/{N_TARGET}' {LOG} | tail -1")
done_n = int(prog.split('/')[0]) if '/' in prog else 0
rate = (en / done_n * 100) if done_n else 0
mark('OK' if rate < 1 else ('주의' if rate < 5 else '이상'),
     'B2 에이전트 실패율', f"{errs} / {prog} ({rate:.2f}%)")

tfiles = sorted(glob.glob(f'{OUT}/timing/day_*.json'))
mtimes = sorted(os.path.getmtime(f) for f in
                sorted(glob.glob(f'{OUT}/metrics/day_*.jsonl')))
durs = [(mtimes[i] - mtimes[i-1]) / 3600 for i in range(1, len(mtimes))]
if len(durs) >= 2:
    slow = durs[-1] / durs[0]
    mark('OK' if slow < 1.5 else ('주의' if slow < 2.5 else '이상'), 'B3 일자별 감속',
         " → ".join(f"{d:.1f}h" for d in durs) + f"  (최근/최초 {slow:.2f}배)")
else:
    mark('정보', 'B3 일자별 감속', " → ".join(f"{d:.1f}h" for d in durs) or '자료 부족')

sf = f'{OUT}/stage1_failures.jsonl'
if os.path.exists(sf):
    nsf = sum(1 for _ in open(sf, encoding='utf-8'))
    # 실패 파일은 전 일자 누적이므로 분모도 누적 agent-day 여야 한다.
    # (현재 일자 진행분으로 나누면 몇 배로 부풀려진다)
    agent_days = sum(sum(1 for _ in open(f, encoding='utf-8'))
                     for f in glob.glob(f'{OUT}/metrics/day_*.jsonl'))
    r_ = 100 * nsf / max(1, agent_days)
    mark('OK' if r_ < 5 else '주의', 'B4 Stage1 재시도',
         f"{nsf:,}건 / {agent_days:,} agent-day ({r_:.2f}%) — 재시도로 복구됨")

ck = sorted(glob.glob(f'{OUT}/checkpoints/done_*.json'))
if ck:
    d = json.load(open(ck[-1], encoding='utf-8'))
    cnt = len(d) if isinstance(d, (list, dict)) else 0
    mark('정보', 'B5 최신 체크포인트', f"{os.path.basename(ck[-1])} · {cnt:,}명")

# ══ C. 데이터 무결성 ═════════════════════════════════════════════════════
print("\n── C. 데이터 무결성 ──")
with driver_session() as s:
    days = [r['d'] for r in s.run(
        "MATCH (pl:Plan) WHERE pl.day>=date('2025-07-14') "
        "RETURN DISTINCT pl.day AS d ORDER BY d")]
    done_days = [str(d) for d in days[:-1]] if alive else [str(d) for d in days]

    dup = s.run("""MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan) WHERE pl.day>=date('2025-07-14')
        WITH a, pl.day AS d, count(pl) AS c WHERE c>1 RETURN count(*) AS n""").single()['n']
    mark('OK' if dup == 0 else '이상', 'C1 하루 중복 Plan', f"{dup}건")

    for d in done_days:
        r = s.run("""MATCH (pl:Plan {day:date($d)}) RETURN count(pl) AS p""", d=d).single()
        gap = N_TARGET - r['p']
        mark('OK' if gap <= N_TARGET * 0.01 else '주의',
             f'C2 {d} Plan 수', f"{r['p']:,} / {N_TARGET:,} (누락 {gap})")

    r = s.run("""MATCH (:Plan)-[i:INCLUDES]->() WHERE i.actual_spent IS NOT NULL
        RETURN sum(CASE WHEN i.actual_spent<0 THEN 1 ELSE 0 END) AS neg,
               sum(CASE WHEN i.actual_spent>1000000 THEN 1 ELSE 0 END) AS huge,
               max(i.actual_spent) AS mx""").single()
    neg, huge, mx = r['neg'] or 0, r['huge'] or 0, r['mx']
    mark('OK' if neg == 0 else '이상', 'C3 음수 결제', f"{neg}건")
    mark('OK' if huge == 0 else '주의', 'C4 100만원 초과 결제',
         f"{huge}건" + (f" (최대 {mx:,}원)" if mx is not None else " (결제 없음)"))

    r = s.run("""MATCH (pl:Plan)-[i:INCLUDES]->() WHERE pl.day>=date('2025-07-14')
        RETURN count(i) AS n, sum(CASE WHEN i.poi_id IS NULL THEN 1 ELSE 0 END) AS nopoi
        """).single()
    mark('정보', 'C5 이벤트 총계', f"{r['n']:,}건")

    if done_days:
        d0 = done_days[-1]
        r = s.run("""MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan {day:date($d)})
            OPTIONAL MATCH (pl)-[i:INCLUDES]->() WHERE coalesce(i.actual_spent,0)>0
            WITH a, count(i) AS c RETURN sum(CASE WHEN c=0 THEN 1 ELSE 0 END) AS zero,
                 count(a) AS tot""", d=d0).single()
        # 오프라인 결제만 세면 온라인으로 쓴 사람이 '무소비'로 잡힌다.
        # (07-18 400명이 전원 소비 10분위였고, 실제로는 온라인 13만원을 썼다)
        _zero = _tot = 0
        for _p in glob.glob(f'{OUT}/metrics/day_{d0}.jsonl'):
            for _l in open(_p, encoding='utf-8'):
                _d = json.loads(_l)
                if _d.get('status') != 'ok':
                    continue
                _tot += 1
                _amt = int(_d.get('cm_today_total_incl_online')
                           or _d.get('cm_today_total') or 0)
                if _amt == 0:
                    _zero += 1
        if _tot:
            z = 100 * _zero / _tot
            mark('OK' if z < 5 else '주의', 'C6 무소비(온라인 포함)',
                 f"{d0} {_zero:,}명 / {_tot:,} ({z:.1f}%)")
        else:
            z = 100 * r['zero'] / max(1, r['tot'])
            mark('정보', 'C6 무소비(오프라인만)', f"{d0} {r['zero']:,}명 ({z:.1f}%)")

    # 기억 — 전날 방문이 기억으로 넘어왔는가
    if len(done_days) >= 2:
        d_prev = done_days[-2]
        r = s.run("""MATCH (pl:Plan {day:date($d)})-[i:INCLUDES]->()
            WHERE i.actual_satisfaction IS NOT NULL RETURN count(i) AS ev""", d=d_prev).single()
        m = s.run("""MATCH (m:Memory {type:'visited'}) WHERE m.day=date($d)
            RETURN count(m) AS n""", d=d_prev).single()
        cov = 100 * m['n'] / max(1, r['ev'])
        mark('OK' if cov > 30 else '주의', 'C7 기억 전환율',
             f"{d_prev} 이벤트 {r['ev']:,} → 기억 {m['n']:,} ({cov:.0f}%)")

    r = s.run("""MATCH (pl:Plan)-[i:INCLUDES]->() WHERE pl.day>=date('2025-07-14')
        AND coalesce(i.actual_spent,0)>0 RETURN count(i) AS n,
        sum(CASE WHEN i.reasoning IS NOT NULL AND i.reasoning<>'' THEN 1 ELSE 0 END) AS rs,
        sum(CASE WHEN i.pick_reason IS NOT NULL AND i.pick_reason<>'' THEN 1 ELSE 0 END) AS pr
        """).single()
    n = max(1, r['n'])
    mark('OK' if 100*r['rs']/n > 95 else '주의', 'C8 reasoning 채움', f"{100*r['rs']/n:.1f}%")
    mark('OK' if 100*r['pr']/n > 70 else '주의', 'C9 pick_reason 채움', f"{100*r['pr']/n:.1f}%")

    # ══ D. 행동 타당성 ═══════════════════════════════════════════════════
    print("\n── D. 행동 타당성 ──")
    per = []
    for d in done_days:
        r = s.run("""MATCH (pl:Plan {day:date($d)})-[i:INCLUDES]->()
            WHERE coalesce(i.actual_spent,0)>0
            RETURN count(DISTINCT pl) AS p, count(i) AS n, sum(i.actual_spent) AS amt,
                   pl.day_type AS dt""", d=d).single()
        if r and r['p']:
            per.append((d, r['dt'], r['amt']//r['p'], r['n']/r['p']))
    # 매 점검마다 묻게 되던 것들 — 건당 금액·요일·공간을 표에 상시 표시한다.
    rows_sp = {r['d']: r for r in s.run("""
        MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
        WHERE pl.day>=date('2025-07-14') AND coalesce(i.actual_spent,0)>0
          AND p.dong_code IS NOT NULL AND a.residence_dong_code_raw IS NOT NULL
        RETURN toString(pl.day) AS d, sum(i.actual_spent) AS amt,
          sum(CASE WHEN p.dong_code=a.residence_dong_code_raw THEN i.actual_spent ELSE 0 END) AS home,
          sum(CASE WHEN left(p.dong_code,5)<>left(a.residence_dong_code_raw,5)
                   THEN i.actual_spent ELSE 0 END) AS og""")}
    print(f"       {'일자':<12}{'요일':<9}{'1인소비':>9}{'건수':>6}{'건당':>8}{'거주동':>8}{'타구':>7}")
    for d, dt, amt, ev in per:
        sp = rows_sp.get(str(d))
        h = f"{100*sp['home']/sp['amt']:>7.1f}%" if sp and sp['amt'] else "      -"
        o = f"{100*sp['og']/sp['amt']:>6.1f}%" if sp and sp['amt'] else "     -"
        print(f"       {str(d):<12}{str(dt):<9}{amt:>9,}{ev:>6.1f}{int(amt/max(ev,.01)):>8,}{h}{o}")
    wd = [p[2] for p in per if p[1] == 'weekday']
    we = [p[2] for p in per if p[1] == 'weekend']
    if wd and we:
        r_ = (sum(we)/len(we)) / (sum(wd)/len(wd))
        mark('OK' if abs(r_-0.735)/0.735 <= .30 else '주의', 'D6 주말/평일 소비비',
             f"{r_:.3f} (BDC 0.735)")
    if len(per) >= 3:
        b0, b1 = per[0][2]/max(per[0][3], .01), per[-1][2]/max(per[-1][3], .01)
        mark('OK' if b1/b0 > 0.6 else '주의', 'D7 건당 금액 표류',
             f"{int(b0):,}원 → {int(b1):,}원 ({b1/b0:.2f}배)")
    if len(per) >= 2:
        vals = [p[2] for p in per]
        sw = max(vals) / min(vals)
        mark('OK' if sw < 1.4 else '주의', 'D1 일별 소비 변동', f"최대/최소 {sw:.2f}배")

    r = s.run("""MATCH (pl:Plan)-[i:INCLUDES]->(p:POI) WHERE pl.day>=date('2025-07-14')
        AND coalesce(i.actual_spent,0)>0
        WITH p, count(i) AS c ORDER BY c DESC LIMIT 1
        RETURN p.name AS nm, c AS top""").single()
    tot_ev = s.run("""MATCH (pl:Plan)-[i:INCLUDES]->() WHERE pl.day>=date('2025-07-14')
        AND coalesce(i.actual_spent,0)>0 RETURN count(i) AS n""").single()['n']
    if r and tot_ev:
        share = 100 * r['top'] / tot_ev
        mark('OK' if share < 2 else '주의', 'D2 최다 POI 쏠림',
             f"{r['nm'][:22]} {r['top']:,}건 ({share:.2f}%)")
    else:
        mark('정보', 'D2 최다 POI 쏠림', "결제 자료 없음(런 초기)")

    hrs = Counter()
    for x in s.run("""MATCH (pl:Plan)-[i:INCLUDES]->() WHERE pl.day>=date('2025-07-14')
        AND coalesce(i.actual_spent,0)>0 AND i.time IS NOT NULL
        RETURN toString(i.time) AS t"""):
        hrs[x['t'][:2]] += 1
    if hrs:
        top_h, top_c = hrs.most_common(1)[0]
        hs = 100 * top_c / sum(hrs.values())
        mark('OK' if hs < 25 else '주의', 'D3 시간대 쏠림',
             f"최다 {top_h}시 {hs:.1f}% · 활동 시간대 {len(hrs)}개")

    r = s.run("""MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
        WHERE pl.day>=date('2025-07-14') AND coalesce(i.actual_spent,0)>0
          AND p.dong_code IS NOT NULL AND a.residence_dong_code_raw IS NOT NULL
        RETURN sum(i.actual_spent) AS t,
          sum(CASE WHEN p.dong_code=a.residence_dong_code_raw THEN i.actual_spent ELSE 0 END) AS home,
          sum(CASE WHEN left(p.dong_code,5)<>left(a.residence_dong_code_raw,5)
                   THEN i.actual_spent ELSE 0 END) AS outgu""").single()
    _t = r['t'] or 0
    mark('정보', 'D4 공간 분포',
         (f"거주동 {100*r['home']/_t:.1f}% · 타구유입 {100*r['outgu']/_t:.1f}%")
         if _t else "결제 자료 없음(런 초기)")

    r = s.run("""MATCH (a:Agent)-[kp:KNOWS_POI]->() WHERE kp.visit_count>=2
        RETURN count(kp) AS re""").single()['re']
    tot_kp = s.run("MATCH ()-[kp:KNOWS_POI]->() RETURN count(kp) AS n").single()['n']
    mark('정보', 'D5 재방문(단골)', f"{r:,} / {tot_kp:,} ({100*r/max(1,tot_kp):.1f}%)")

# ══ E. 백업 ══════════════════════════════════════════════════════════════
print("\n── E. 백업 ──")
# 복구 재개로 런 디렉터리가 갈리면(out_X, out_X_r2) 한쪽만 세어 헛경보가 난다.
# 완료 일자에 해당하는 원장을 모든 out_* 에서 찾는다.
ev = sorted({f for d in done_days
             for f in glob.glob(f'/data/exp001/out_*/events_{d}.jsonl')})
mark('OK' if len(ev) >= len(done_days) else '주의', 'E1 일자별 원장',
     f"{len(ev)}개 / 완료 {len(done_days)}일")
for f in ev:
    n = sum(1 for _ in open(f, encoding='utf-8'))
    print(f"       {os.path.basename(f)}  {n:,}건  {os.path.getsize(f)/1e6:.1f}MB")
if ev:
    d = json.loads(open(ev[-1], encoding='utf-8').readline())
    need = ['poi_dong', 'res_dong', 'work_dong_name', 'time', 'trigger']
    miss = [k for k in need if k not in d]
    mark('OK' if not miss else '주의', 'E2 원장 필드',
         "전체 포함" if not miss else f"누락 {miss}")

# ══ F. 기능 활성 · 데이터 정합 ═══════════════════════════════════════════
# "조용히 꺼진 채 도는" 사고를 잡기 위한 섹션.
#   - output/stats 누락으로 광역상권이 60시간 동안 0개였던 일
#   - 야간 정산 누락으로 하루치 기억·KNOWS_POI 가 통째로 빠진 일
#   - 7자리/8자리 코드를 조인해 "직장 소비 0%"로 오독한 일
print("\n── F. 기능 활성 · 데이터 정합 ──")

try:
    sys.path.insert(0, '/data/exp001_repo/scripts')
    from sim import mobility
    hb = mobility._load()
    nh, nc = len(hb['all_hubs']), len(hb['centroids'])
    mark('OK' if nh >= 100 else '이상', 'F1 광역상권 허브',
         f"풀 {nh}개 · 좌표 {nc}개" + ("" if nh else "  ← suggest_hubs 항상 0개"))
    if nh:
        mark('OK' if 100*nc/nh >= 95 else '주의', 'F2 허브 좌표 커버리지',
             f"{100*nc/nh:.0f}% — 없는 동은 자치구 중심 대체(거리 0km 왜곡)")
except Exception as e:
    mark('이상', 'F1 광역상권 허브', f"로드 실패 {e}")

with driver_session() as s:
    rows = list(s.run("""MATCH (pl:Plan) WHERE pl.day>=date('2025-07-14')
        WITH DISTINCT pl.day AS d ORDER BY d
        CALL { WITH d MATCH (:Plan {day:d})-[i:INCLUDES]->()
               WHERE i.actual_satisfaction IS NOT NULL RETURN count(i) AS ev }
        CALL { WITH d MATCH (m:Memory {type:'visited'}) WHERE m.day=d RETURN count(m) AS mem }
        CALL { WITH d MATCH ()-[kp:KNOWS_POI]->() WHERE kp.last_visit=d RETURN count(kp) AS kn }
        RETURN d, ev, mem, kn"""))
    bad = [r for r in rows[:-1] if r['ev'] > 100 and (r['mem'] == 0 or r['kn'] == 0)]
    mark('OK' if not bad else '이상', 'F3 야간 정산 누락',
         "없음" if not bad else " · ".join(
             f"{r['d']}(이벤트 {r['ev']:,}→기억 {r['mem']}·단골 {r['kn']})" for r in bad))

    r = s.run("""MATCH (a:Agent) WITH a LIMIT 2000
        RETURN collect(DISTINCT size(toString(a.residence_dong_code_raw)))[0..3] AS res,
               collect(DISTINCT size(toString(a.workplace_dong_code_raw)))[0..3] AS wrk""").single()
    pl_ = sorted(x['L'] for x in s.run(
        "MATCH (p:POI) WHERE p.dong_code IS NOT NULL RETURN DISTINCT size(toString(p.dong_code)) AS L LIMIT 3"))
    mark('OK' if set(r['res']) & set(pl_) else '이상', 'F4 조인 키 자릿수',
         f"거주 {r['res']} · 직장 {r['wrk']} · POI {pl_}"
         + ("" if set(r['wrk']) & set(pl_) else "  ← 직장은 체계가 달라 직접 조인 불가(이름 대조)"))

# ══ G. 그래프 엣지 적재 ═══════════════════════════════════════════════════
# 런타임에 쓰이는 엣지가 일자별로 제대로 채워지는지. 노드만 생기고 엣지가 비면
# 컨텍스트 빌더가 빈 손으로 돌아 조용히 품질이 떨어진다.
print("\n── G. 그래프 엣지 적재 ──")
with driver_session() as s:
    def one(q, **kw):
        r = s.run(q, **kw).single()
        return (r[0] if r else 0) or 0

    n_plan = one("MATCH (p:Plan) RETURN count(p)")
    n_hp = one("MATCH (:Agent)-[r:HAS_PLAN]->(:Plan) RETURN count(r)")
    mark('OK' if n_hp == n_plan else '이상', 'G1 HAS_PLAN ↔ Plan',
         f"{n_hp:,} / {n_plan:,}" + ("" if n_hp == n_plan else "  ← 고아 Plan"))

    n_mem = one("MATCH (m:Memory) RETURN count(m)")
    n_rem = one("MATCH (:Agent)-[r:REMEMBERS]->(:Memory) RETURN count(r)")
    mark('OK' if n_rem >= n_mem else '주의', 'G2 REMEMBERS ↔ Memory', f"{n_rem:,} / {n_mem:,}")

    n_vis = one("MATCH (m:Memory {type:'visited'}) RETURN count(m)")
    n_ab = one("MATCH (m:Memory {type:'visited'})-[:ABOUT_POI]->(:POI) RETURN count(m)")
    cov = 100 * n_ab / max(1, n_vis)
    mark('OK' if cov >= 99 else '주의', 'G3 ABOUT_POI 커버리지',
         f"{n_ab:,} / {n_vis:,} ({cov:.1f}%)")

    print("       일자별 런타임 엣지")
    for x in s.run("""MATCH (pl:Plan) WHERE pl.day>=date('2025-07-14')
        WITH DISTINCT pl.day AS d ORDER BY d
        CALL { WITH d MATCH (:Plan {day:d})-[i:INCLUDES]->() RETURN count(i) AS inc }
        CALL { WITH d MATCH (m:Memory) WHERE m.day=d RETURN count(m) AS mem }
        CALL { WITH d MATCH (c:Conversation) WHERE c.day=d RETURN count(c) AS conv }
        CALL { WITH d MATCH (st:State) WHERE st.day=d RETURN count(st) AS stt }
        RETURN d, inc, mem, conv, stt"""):
        print(f"       {str(x['d'])}  INCLUDES {x['inc']:>7,} · Memory {x['mem']:>7,} · "
              f"Conversation {x['conv']:>6,} · State {x['stt']:>6,}")

    r = s.run("""MATCH (pl:Plan)-[i:INCLUDES]->() WHERE pl.day>=date('2025-07-14')
        RETURN count(i) AS n,
          sum(CASE WHEN i.category IS NULL THEN 1 ELSE 0 END) AS c,
          sum(CASE WHEN i.time IS NULL THEN 1 ELSE 0 END) AS t,
          sum(CASE WHEN i.anchor IS NULL THEN 1 ELSE 0 END) AS a,
          sum(CASE WHEN i.actual_spent IS NULL THEN 1 ELSE 0 END) AS sp""").single()
    n = max(1, r['n'])
    worst = max(r['c'], r['t'], r['a'])
    mark('OK' if 100*worst/n < 1 else '주의', 'G4 INCLUDES 필수속성 결측',
         f"category {100*r['c']/n:.2f}% · time {100*r['t']/n:.2f}% · anchor {100*r['a']/n:.2f}%")

    st_days = one("MATCH (st:State) WHERE st.day>=date('2025-07-14') RETURN count(DISTINCT st.day)")
    mark('OK' if st_days > 0 else '주의', 'G5 State 일자 적재', f"{st_days}일")

# ══ H. BOK 유효성 검증 ════════════════════════════════════════════════════
# 한국은행 이슈노트 2026-13 실측과의 대조. 실측값은 **판정에만** 쓰고 시뮬 입력으로는
# 넣지 않는다(순환 방지). 정책 구간에서만 산출된다.
REF_MPC = 0.21                                            # 30항 1차
REF_MPC_Q = {1: .27, 2: .245, 3: .185, 4: .21, 5: .165}   # 31항·그림18 ◆
REF_DEP = 2.73                                            # 표1 4주 76.4% → 일평균
REF_T3 = {"음식점": 46.0, "마트식료품": 21.6, "의료": 6.2,
          "미용": 5.1, "학원": 4.0, "약국": 3.9}            # 표3 1차
REF_ITEM_MPC = {"내구재": .48, "준내구재": .43, "여가": .41, "개인미용": .35,
                "숙박음식": .29, "비내구재": .13, "병원": .09, "학원": .04}  # 그림19 막대

def _vd(obs, ref, tol=.25):
    """실측 대비 상대오차로 판정. tol 이내 OK, 2배 이내 주의, 그 밖 이상."""
    if not ref:
        return 'OK' if not obs else '주의'
    d = abs(obs - ref) / ref
    return 'OK' if d <= tol else ('주의' if d <= tol*2 else '이상')


_pol_files = sorted(glob.glob(f'{OUT}/metrics/day_*.jsonl'))
_used, _mpc, _mq, _dep = [], [], defaultdict(list), defaultdict(list)
for _p in _pol_files:
    for _l in open(_p, encoding='utf-8'):
        _d = json.loads(_l)
        if _d.get('status') != 'ok':
            continue
        _a = int(_d.get('cm_policy_allocated_total') or 0)
        _w = int(_d.get('cm_policy_wallet_available') or 0) or \
             int(_d.get('cm_intended_grant_today') or 0)
        # 소진율의 분모는 '수령자 전원'이다. 사용한 사람만 평균 내면 사용률(약 40%)의
        # 역수만큼 부풀려진다(6.47% vs 실제 약 2.6%). 실측 2.73%/일도 전원 기준이다.
        if _w > 0:
            _dep['40만' if _w >= 350000 else '30만' if _w >= 250000 else '15만'].append(100*_a/_w)
        if _a <= 0:
            continue
        _used.append(_d)
        _v = _d.get('cm_mpc_new_share')
        if _v is not None:
            _mpc.append((float(_v), _a))
            _dc = _d.get('spend_decile')
            if _dc:
                _mq[min(5, max(1, (int(_dc)+1)//2))].append((float(_v), _a))

if not _mpc:
    print("\n── H. BOK 유효성 검증 ──\n  (무정책 구간 — 정책 지표 없음)")
else:
    print("\n── H. BOK 유효성 검증 ──")
    _tw = sum(b for _, b in _mpc)
    _m = sum(a*b for a, b in _mpc) / _tw
    mark(_vd(_m, REF_MPC), 'H1 전체 MPC',
         f"{_m:.3f} (실측 {REF_MPC}) n={len(_mpc):,}")
    _vals = {}
    for _k in sorted(_mq):
        _t = sum(b for _, b in _mq[_k])
        _vals[_k] = sum(a*b for a, b in _mq[_k]) / _t
        mark(_vd(_vals[_k], REF_MPC_Q[_k]), f'H2 {_k}분위 MPC',
             f"{_vals[_k]:.3f} (실측 {REF_MPC_Q[_k]}) n={len(_mq[_k]):,}")
    if 1 in _vals and 5 in _vals:
        _gap = _vals[1] - _vals[5]
        mark('OK' if abs(_gap-0.105) <= .05 else '주의', 'H3 1−5분위 격차',
             f"{_gap:+.3f} (실측 +0.105)")
    if _dep:
        WT = {'40만': 400000, '30만': 300000, '15만': 150000}
        _da = sum(st.mean(_dep[t])*len(_dep[t])*WT[t] for t in _dep) / \
              sum(len(_dep[t])*WT[t] for t in _dep)
        mark(_vd(_da, REF_DEP), 'H4 일간 소진율',
             f"{_da:.2f}%/일 (실측 {REF_DEP}) · " +
             " ".join(f"{t} {st.mean(_dep[t]):.2f}" for t in ('40만', '30만', '15만') if t in _dep))

    # 업종·품목은 결제원장에서 (그래프가 다음 런에 덮여도 재현 가능한 경로)
    sys.path.insert(0, '/data/exp001')
    from item_map import to_item, to_t3   # noqa: E402
    _t3, _imp, _C, _bad, _n = defaultdict(float), defaultdict(lambda: [0.0, 0.0]), 0.0, 0, 0
    for _p in sorted(glob.glob(f'{OUT}/events_*.jsonl')):
        for _l in open(_p, encoding='utf-8'):
            _e = json.loads(_l)
            _amt = float(_e.get('amt') or 0)
            if _amt <= 0:
                continue
            _n += 1
            _sp = _e.get('sp')
            _c = 0.0
            if _sp and _sp not in ('{}', 'null'):
                try:
                    _c = float(sum(json.loads(_sp).values()))
                except Exception:
                    _c = 0.0
            _c = max(0.0, min(_amt, _c))
            if _c <= 0:
                continue
            _C += _c
            if _e.get('elig') is False:
                _bad += 1
            _ex = _e.get('ex')
            if _ex is None:
                _ex = 0.0 if _e.get('wba') else _amt
            _ex = max(0.0, min(_amt, float(_ex))) * (_c/_amt)
            _k = to_t3(_e.get('l1'), _e.get('sub'))
            if _k:
                _t3[_k] += _c
            _it = to_item(_e.get('l1'), _e.get('sub'))
            _imp[_it][0] += _c
            _imp[_it][1] += _ex
    if _C > 0:
        _ng = []
        for _k, _r in REF_T3.items():
            _o = 100*_t3.get(_k, 0)/_C
            if abs(_o-_r)/_r > .40:
                _ng.append(f"{_k} {_o:.1f}({_r})")
        mark('OK' if not _ng else '주의', 'H5 업종 구성 (표3)',
             "6개 모두 근접" if not _ng else "이탈 " + " · ".join(_ng))
        _hit = []
        for _k, _r in REF_ITEM_MPC.items():
            _cc, _xx = _imp[_k]
            if _cc > 0 and abs(_xx/_cc - _r)/_r <= .50:
                _hit.append(_k)
        mark('OK' if len(_hit) >= 5 else '주의', 'H6 품목별 MPC (그림19)',
             f"{len(_hit)}/8 근접 — {' '.join(_hit[:5])}")
        mark('OK' if _bad == 0 else '이상', 'H7 사용처 준수율',
             f"{100*(1-_bad/max(1,_n)):.2f}% (실측 100)")

print(f"\n{'='*78}\n  [OK] {T['OK']} · [주의] {T['주의']} · [이상] {T['이상']}\n{'='*78}")
