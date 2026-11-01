"""시민 인터뷰용 경험 기록 — 실행 결과 묶음의 dossier.jsonl 에서 사람마다 필요한 것만 뽑는다.

    python web/tools/extract_dossier.py --data <데이터 폴더>            # 모든 정책 실험
    python web/tools/extract_dossier.py --data <데이터 폴더> --only p017_seoul_emergency_living

dossier 는 시뮬레이션이 사람 단위로 남긴 기록이다. 시뮬레이션이 실제로 쓴 프로필(소득·생활 방식),
날마다 들른 가게·메뉴·쓴 돈·만족도, 그때 모델이 밝힌 이유, 들은 이야기가 들어 있다.
인터뷰는 이걸 보고 '그 사람이 겪은 일'로 답한다. 정책 있는 쪽(on)만 뽑는다 — 인터뷰는 그쪽 시민에게만 묻는다.

읽는 곳: 실험 폴더의 _sync.json 이 가리키는 결과 zip(<데이터 폴더>/zips/...) 안의 <묶음>/on/dossier.jsonl
쓰는 곳: <데이터 폴더>/experiments/<실험>/_console/dossier/<시민 id>.json.gz  (+ _index.json)
원본 zip 의 sha256 앞부분을 _index.json 에 적어 둔다. 같은 zip 이면 다시 뽑지 않는다.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

SCHEMA = 2
# 모델 내부 수치("만족 0.88", "avg_sat=0.86", "(0.35km)")와 태그("[lifestyle]")는 사람이 하는 말이 아니다
INTERNAL = [
    re.compile(r"\[[a-z_]+\]\s*"),
    re.compile(r"\(?\b[a-z_]+\s*=\s*-?[\d.]+\)?"),
    re.compile(r"\(?(?:만족(?:도)?|피로(?:도)?|에너지|기분|중요도)\s*-?0?\.\d+\)?"),
    re.compile(r"\((?:거리\s*)?[\d.]+\s*km\)"),
    re.compile(r"\s*\((?:known|new)\)"),
]
# 다른 시민 id("AGT_11110515_F_40대_001과의 약속")는 사람이 쓰는 말이 아니다
AGENT_REF = re.compile(r"AGT_[0-9A-Za-z가-힣_]+?_\d{3}")


RELATION = r"(남편|아내|배우자|친구|동료|이웃|언니|누나|형|오빠|동생|어머니|아버지|엄마|아빠|자녀|딸|아들|가족)"
PARTICLE = {"와": ("과", "와"), "과": ("과", "와"), "를": ("을", "를"), "을": ("을", "를"),
            "는": ("은", "는"), "은": ("은", "는"), "가": ("이", "가"), "이": ("이", "가")}


def _josa(word: str, particle: str) -> str:
    if particle not in PARTICLE:
        return word + particle
    last = word[-1]
    batchim = "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 != 0
    return word + PARTICLE[particle][0 if batchim else 1]


def people(text: str) -> str:
    """다른 시민 id 를 사람 말로: '남편 AGT_…와' → '남편과', 그 밖의 id → '지인'."""
    text = re.sub(RELATION + r"\s*(?:은|는|인)?\s*\(?" + AGENT_REF.pattern + r"\)?([와과를을는은가이])?",
                  lambda m: _josa(m.group(1), m.group(2) or "") if m.group(2) else m.group(1), text)
    return AGENT_REF.sub("지인", text).replace("지인와", "지인과").replace("지인를", "지인을").replace("지인는", "지인은").replace("지인가", "지인이")


# 이런 말이 든 문장은 모델 내부 계산이라 통째로 뺀다(숫자만 지우면 "봄새에서 를 기록했고" 같은 구멍이 남는다)
INTERNAL_SENTENCE = re.compile(
    r"\d\.\d|D-\d|[A-Z]_[A-Za-z]{2}\d|[A-Za-z]{3,}|=|메모리|친밀도|반복 회피|정책\s?지갑|이벤트 설명|후보|기록했|기록하고|"
    r"성향(?:이|을|과)? 반영|성향과 부합|수혜 범위|판단한다|D-|에너지|피로도|중요도")


def clean(text: object, limit: int = 220) -> str:
    raw = people(str(text or ""))
    sentences = re.split(r"(?<=[.!?])\s+|(?<=다\.)", raw)
    kept = [s for s in sentences if s.strip() and not INTERNAL_SENTENCE.search(s)]
    out = " ".join(kept)
    for pattern in INTERNAL:
        out = pattern.sub("", out)
    out = re.sub(r"\s+", " ", out).strip(" ,·")
    return out[:limit]


def as_list(value: object) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def compact(record: dict) -> dict:
    profile = record.get("profile") or {}
    keep = ("age", "age_group", "gender", "job", "life_stage", "income_level", "spending_tendency", "lifestyle",
            "residence_dong", "gu", "workplace_dong", "commute_min", "s_daily_wd", "spending_top_wd_json")
    visits, heard = [], []
    for memory in record.get("memories") or []:
        if memory.get("type") == "visited":
            # 기억 id 에 가게 id 가 들어 있다: mem_vis_<시민>_<가게>_<날짜>_<순서> — 영수증과 짝짓는 열쇠
            poi = None
            match = re.match(r"mem_vis_.+?_((?:C|R|P|S)_[A-Za-z0-9]+)_\d{4}-\d{2}-\d{2}_\d+$", str(memory.get("id") or ""))
            if match:
                poi = match.group(1)
            visits.append({
                "day": memory.get("day"), "poi_id": poi, "store": memory.get("store"), "menu": memory.get("menu"),
                "category": memory.get("category"), "spent": memory.get("spent"), "paid_policy": bool(memory.get("paid_policy")),
                "satisfaction": memory.get("satisfaction"), "importance": memory.get("importance"),
                "why": clean(memory.get("why")), "pick_why": clean(memory.get("pick_why"), 160),
            })
        elif memory.get("type") == "rumor":
            heard.append({"day": memory.get("day"), "text": clean(memory.get("summary"), 200), "importance": memory.get("importance")})
    plans = []
    for plan in record.get("plans") or []:
        items = []
        for item in plan.get("items") or []:
            if item.get("category") == "집":
                continue
            items.append({
                "time": str(item.get("time") or "")[:5], "place": item.get("poi"), "category": item.get("category"),
                "intent": clean(item.get("intent"), 60), "with": len(as_list(item.get("with_agents"))),
            })
        if items:
            plans.append({"day": plan.get("day"), "items": items})
    return {
        "schema": SCHEMA, "aid": record.get("aid"), "window": record.get("window"),
        "profile": {key: profile.get(key) for key in keep if profile.get(key) not in (None, "")},
        "visits": visits, "heard": heard, "plans": plans,
    }


def sha_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def extract(data: Path, experiment: Path, force: bool) -> str:
    sync = experiment / "_sync.json"
    if not sync.is_file():
        return "결과 zip 기록(_sync.json)이 없어 건너뜀"
    meta = json.loads(sync.read_text(encoding="utf-8"))
    zpath = data / "zips" / meta.get("zip", "")
    if not meta.get("zip") or not zpath.is_file():
        return f"결과 zip 이 없어 건너뜀: {zpath}"
    out = experiment / "_console" / "dossier"
    index_path = out / "_index.json"
    source = sha_prefix(zpath)
    if not force and index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("source_sha256_16") == source and index.get("schema") == SCHEMA:
            return f"이미 최신 ({index.get('agents')}명)"
    with zipfile.ZipFile(zpath) as archive:
        members = [n for n in archive.namelist() if n.endswith("/on/dossier.jsonl")]
        if not members:
            return "zip 에 on/dossier.jsonl 이 없어 건너뜀"
        out.mkdir(parents=True, exist_ok=True)
        count, size = 0, 0
        with archive.open(members[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                if not line.strip():
                    continue
                record = compact(json.loads(line))
                aid = str(record.get("aid") or "")
                if not re.fullmatch(r"[A-Za-z0-9_가-힣]+", aid):
                    continue
                body = gzip.compress(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), mtime=0)
                (out / f"{aid}.json.gz").write_bytes(body)
                count += 1
                size += len(body)
    index_path.write_text(json.dumps({"schema": SCHEMA, "source_zip": meta["zip"], "member": members[0],
                                      "source_sha256_16": source, "agents": count, "bytes": size},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return f"{count}명, {size / 1e6:.1f}MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="시민 인터뷰용 경험 기록 뽑기")
    parser.add_argument("--data", required=True, help="experiments·zips 가 들어 있는 데이터 폴더")
    parser.add_argument("--only", action="append", help="이 실험만 (여러 번 줄 수 있음)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 뽑는다")
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    for experiment in sorted((data / "experiments").iterdir()):
        if not experiment.is_dir() or (args.only and experiment.name not in args.only):
            continue
        print(f"{experiment.name}: {extract(data, experiment, args.force)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
