"""군집별 찬반 분포 — 사는 구 · 흡연 여부 · 나이대 · 성별 · 주 이용 시설. 시민 ID(AGT_동코드_성별_나이대_번호)에서 군집을 읽는다."""
import json, sys, io, random, collections
from pathlib import Path

D = Path(sys.argv[1]); ph = sys.argv[2]
GU = {"11350": "노원구", "11650": "서초구", "11710": "송파구"}
rows = [json.loads(l) for l in io.open(D / f"{ph}.jsonl", encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("answer")]


def clusters(r):
    _, dong, sex, age, _ = r["aid"].split("_")
    a = age.replace("대이상", "대+")
    ab = "20~30대" if a in ("20대", "30대") else "40~50대" if a in ("40대", "50대") else "60대 이상" if a in ("60대", "70대+") else a
    return {"사는 구": GU.get(dong[:5], dong[:5]), "흡연 여부": {"smoker": "흡연자", "non_smoker": "비흡연자"}.get(r["smoking_status"], "모름"),
            "나이대": ab, "성별": "남성" if sex == "M" else "여성",
            "주 이용 시설": "당구장" if r.get("billiard", 0) > r.get("golf", 0) else "골프연습장"}


out = collections.defaultdict(lambda: collections.defaultdict(list))
for r in ok:
    for k, v in clusters(r).items():
        out[k][v].append(r["answer"]["choice"])
rnd = random.Random(7)
res = {}
for k, groups in out.items():
    res[k] = {}
    for v, ch in sorted(groups.items()):
        n = len(ch); sup = 100 * ch.count("찬성") / n; opp = 100 * ch.count("반대") / n; unk = 100 * ch.count("모름") / n
        bs = sorted(100 * sum(ch[rnd.randrange(n)] == "찬성" for _ in range(n)) / n for _ in range(2000))
        res[k][v] = dict(n=n, 찬성=round(sup, 1), 반대=round(opp, 1), 모름=round(unk, 1), ci_찬성=[round(bs[50], 1), round(bs[1949], 1)])
(D / f"clusters_{ph}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for k, g in res.items():
    print(k, " | ".join(f"{v}(n={x['n']}) 찬성 {x['찬성']}% [{x['ci_찬성'][0]},{x['ci_찬성'][1]}] 반대 {x['반대']}%" for v, x in g.items()))
