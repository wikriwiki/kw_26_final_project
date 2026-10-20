"""제출용 AI API 호출 로그 묶음을 만든다.

    python -m scripts.ai_call_log_export                 # 기능별 최신 성공 1건
    python -m scripts.ai_call_log_export --per-feature 3
    python -m scripts.ai_call_log_export --features 1대1_인터뷰 보고서_해설

``output/ai_call_logs/<기능명>.jsonl`` 을 읽어 가이드의 제출 형식으로 바꾼다.

- 2026-10-14 00:00 KST 이후에 요청한 건만 쓴다 (가이드: 그 이전 호출은 인정하지 않는다).
- 응답을 못 받은 건(오류)은 쓰지 않는다. 응답 본문이 없으면 확인할 것이 없다.
- ``연결_점검`` 은 기능이 아니라서 기본으로 뺀다.
- 국외 AI 엔드포인트(``ai_call_log.FOREIGN_AI_HOSTS``)로 간 건이 하나라도 있으면 **멈춘다.**
  국내 AI 트랙은 100% 국내 AI 연계 기업 모델이어야 한다. 그런 기록을 몰래 빼고
  묶으면 제출물이 실제 구성을 숨기게 된다.
- 기능마다 ``<기능명>.json`` (UTF-8, 들여쓰기) 하나를 만들고 전부 zip 하나로 묶는다.
- 쓰기 전에 키 모양 문자열을 한 번 더 가리고, 남은 것이 있으면 멈춘다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from scripts import ai_call_log

CUTOFF = datetime.fromisoformat("2026-10-14T00:00:00+09:00")
EXCLUDED_FEATURES = {"연결_점검"}
_LEFTOVER_SECRET = re.compile(r"Bearer\s+\S{8,}|\bflp_[A-Za-z0-9]{8,}|\bsk-[A-Za-z0-9_\-]{16,}|\bAIza[0-9A-Za-z_\-]{30,}")


def load(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  ! {path.name}:{number} 읽지 못함 ({exc})", file=sys.stderr)
    return rows


def usable(row: dict) -> bool:
    request, response = row.get("request") or {}, row.get("response") or {}
    try:
        requested = datetime.fromisoformat(request["requested_at"])
    except (KeyError, ValueError):
        return False
    return requested >= CUTOFF and response.get("body") is not None and not response.get("error")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=Path, default=None, help="기본: output/ai_call_logs (AI_CALL_LOG_DIR)")
    parser.add_argument("--out", type=Path, default=None, help="zip 경로 (기본: <로그폴더>/submission/...zip)")
    parser.add_argument("--per-feature", type=int, default=1, help="기능별로 넣을 건수 (최신순). 가이드 최소 1건")
    parser.add_argument("--features", nargs="*", help="이 기능만 묶는다 (파일명에서 .jsonl 을 뺀 이름)")
    args = parser.parse_args(argv)

    log_dir = args.log_dir or ai_call_log.log_dir()
    files = sorted(log_dir.glob("*.jsonl"))
    if args.features:
        files = [f for f in files if f.stem in set(args.features)]
        missing = set(args.features) - {f.stem for f in files}
        if missing:
            print(f"로그가 없는 기능: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
    else:
        files = [f for f in files if f.stem not in EXCLUDED_FEATURES]
    if not files:
        print(f"묶을 로그가 없습니다: {log_dir}", file=sys.stderr)
        return 1

    picked: dict[str, list[dict]] = {}
    foreign: list[str] = []
    for path in files:
        rows = load(path)
        for row in rows:
            endpoint = (row.get("request") or {}).get("endpoint", "")
            if ai_call_log.is_foreign_ai_endpoint(endpoint):
                foreign.append(f"{path.stem}: {urlparse(endpoint).hostname}")
        good = [r for r in rows if usable(r)]
        good.sort(key=lambda r: r["request"]["requested_at"], reverse=True)
        print(f"- {path.stem}: 전체 {len(rows)}건, 제출 가능 {len(good)}건")
        if good:
            picked[path.stem] = list(reversed(good[: args.per_feature]))
    if foreign:
        print("\n국외 AI 엔드포인트로 간 호출이 있습니다. 묶지 않고 멈춥니다:", file=sys.stderr)
        for item in sorted(set(foreign)):
            print(f"  ! {item}", file=sys.stderr)
        print("해당 기능을 국내 모델로 바꾸고 로그 파일을 새로 만든 뒤 다시 실행하십시오.", file=sys.stderr)
        return 2
    empty = [f.stem for f in files if f.stem not in picked]
    if empty:
        print(f"\n10/14 이후 성공한 호출이 없는 기능: {', '.join(empty)}", file=sys.stderr)
        return 1

    documents: dict[str, str] = {}
    for feature, rows in picked.items():
        document = {
            "feature": feature,
            "timezone": "KST (UTC+09:00)",
            "record_count": len(rows),
            "records": [ai_call_log.mask_secrets(r) for r in rows],
        }
        text = json.dumps(document, ensure_ascii=False, indent=2)
        if _LEFTOVER_SECRET.search(text):
            print(f"{feature}: 키 모양 문자열이 남아 있어 멈춥니다. 원본 로그를 확인하십시오.", file=sys.stderr)
            return 3
        documents[feature] = text

    stamp = datetime.now(ai_call_log.KST).strftime("%Y%m%d_%H%M")
    out = args.out or log_dir / "submission" / f"AI_API_호출로그_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for feature, text in documents.items():
            bundle.writestr(f"{feature}.json", text.encode("utf-8"))
    print(f"\n제출 파일: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
