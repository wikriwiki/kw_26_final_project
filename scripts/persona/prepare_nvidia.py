"""
scripts/persona/prepare_nvidia.py — NVIDIA Nemotron-Personas-Korea 실데이터 준비
================================================================================
HuggingFace `nvidia/Nemotron-Personas-Korea`(1.0M행, CC BY 4.0)를 **스트리밍**으로
받아 `province` 필터(기본 '서울') 후, 페르소나 빌더가 읽는
`data/personas/nvidia_seoul_full.json(.jsonl)` 로 저장한다. **1회성 준비 스크립트.**

스트리밍이라 1M행을 메모리에 올리지 않으며, 받은 만큼 바로 디스크에 흘려쓴다.
저장 후에는 `load_nvidia_seoul()` 가 sample 대신 이 full 파일을 자동 우선 사용.

사용:
  python scripts/persona/prepare_nvidia.py                 # 서울 전체 → full.json
  python scripts/persona/prepare_nvidia.py --jsonl         # 서울 전체 → full.jsonl(대용량 권장)
  python scripts/persona/prepare_nvidia.py --province 경기 # 다른 시·도
  python scripts/persona/prepare_nvidia.py --all           # 전국 1M 전체
  python scripts/persona/prepare_nvidia.py --limit 2000    # 스모크 테스트

요구: pip install "datasets>=2.14"   (네트워크 필요; HF 토큰 불필요 — 공개 데이터)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASET_ID = "nvidia/Nemotron-Personas-Korea"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "personas"

# 데이터셋 스키마 26개 컬럼 (그대로 보존)
FIELDS = (
    "uuid", "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "family_persona", "persona",
    "cultural_background", "skills_and_expertise", "skills_and_expertise_list",
    "hobbies_and_interests", "hobbies_and_interests_list",
    "career_goals_and_ambitions", "sex", "age", "marital_status",
    "military_status", "family_type", "housing_type", "education_level",
    "bachelors_field", "occupation", "district", "province", "country",
)


# ---------------------------------------------------------------------------
# 순수 함수 (네트워크 없이 단위 테스트 가능)
# ---------------------------------------------------------------------------
def keep_record(rec: dict, province: str, keep_all: bool) -> bool:
    """province 필터. keep_all=True 면 전부 통과. '서울특별시' 등 변형도 허용."""
    if keep_all:
        return True
    p = (rec.get("province") or "").strip()
    return p == province or p.startswith(province)


def project_record(rec: dict) -> dict:
    """26개 스키마 필드만 추출 (datasets 내부 메타 제거, 순서 고정)."""
    return {k: rec.get(k) for k in FIELDS}


# ---------------------------------------------------------------------------
# 데이터셋 스트리밍 로드 (lazy import — datasets 없으면 친절히 안내)
# ---------------------------------------------------------------------------
def _iter_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.stderr.write(
            "[prepare_nvidia] `datasets` 미설치. 다음으로 설치하세요:\n"
            "    pip install \"datasets>=2.14\"\n"
        )
        raise SystemExit(2)
    # streaming=True → 1M행을 메모리에 올리지 않고 순차 iterate
    return load_dataset(DATASET_ID, split="train", streaming=True)


# ---------------------------------------------------------------------------
# 메인 준비 루틴
# ---------------------------------------------------------------------------
def prepare(province: str = "서울", keep_all: bool = False, limit: int = 0,
            jsonl: bool = False, out: Path | None = None,
            log_every: int = 50_000) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out is None:
        name = "nvidia_seoul_full.jsonl" if jsonl else "nvidia_seoul_full.json"
        out = OUT_DIR / name

    ds = _iter_dataset()
    seen: set[str] = set()
    n_in = n_out = n_dup = 0
    scope = "전국" if keep_all else province

    print(f"[prepare_nvidia] {DATASET_ID} 스트리밍 시작 → 필터: {scope}")

    if jsonl:
        f = out.open("w", encoding="utf-8")
        write = lambda row: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        records: list[dict] = []
        write = records.append
        f = None

    try:
        for rec in ds:
            n_in += 1
            if n_in % log_every == 0:
                print(f"  …{n_in:,}행 스캔, {n_out:,}건 수집")
            if not keep_record(rec, province, keep_all):
                continue
            uuid = rec.get("uuid")
            if uuid and uuid in seen:
                n_dup += 1
                continue
            if uuid:
                seen.add(uuid)
            write(project_record(rec))
            n_out += 1
            if limit and n_out >= limit:
                print(f"  --limit {limit} 도달, 중단")
                break
    finally:
        if f is not None:
            f.close()

    if not jsonl:
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(f"[prepare_nvidia] 완료: {n_out:,}건 (스캔 {n_in:,}, 중복 {n_dup:,}) "
          f"→ {out}  ({size_mb:.1f} MB)")
    if n_out == 0:
        print(f"  ⚠️ 0건 — province 값 확인 필요(현재 '{province}'). "
              f"예: 서울/경기/부산/인천/대구/광주/대전/울산/세종/강원/충북/충남/"
              f"전북/전남/경북/경남/제주")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="NVIDIA Nemotron-Personas-Korea 실데이터 준비")
    ap.add_argument("--province", default="서울", help="시·도 필터값 (기본 '서울')")
    ap.add_argument("--all", action="store_true", help="전국 1M 전체 (필터 끔)")
    ap.add_argument("--jsonl", action="store_true", help="JSONL 라인 출력 (대용량 권장)")
    ap.add_argument("--limit", type=int, default=0, help="수집 상한 (스모크 테스트)")
    ap.add_argument("--out", type=Path, default=None, help="출력 경로 직접 지정")
    args = ap.parse_args()

    prepare(province=args.province, keep_all=args.all, limit=args.limit,
            jsonl=args.jsonl, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
