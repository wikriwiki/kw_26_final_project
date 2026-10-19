"""3주 A/B 실행기: 엔진 소스 지문과 실행 설정을 준비 때 고정하고, 이어 돌릴 때 같은지 확인한다.

    python tools/ab3w_engine_record.py <engine.json>            # 기록
    python tools/ab3w_engine_record.py --check <engine.json>    # 확인(다르면 종료 코드 1)
정책 전 주와 두 갈래가 같은 엔진·같은 설정으로 돌아야 짝이다.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sim"))
from experience_provenance import execution_fingerprint, source_fingerprint  # noqa: E402

check = sys.argv[1] == "--check"
path = Path(sys.argv[-1])
now = {"source_fingerprint": source_fingerprint(), "execution_fingerprint": execution_fingerprint()}
if check:
    old = json.loads(path.read_text(encoding="utf-8"))
    bad = [k for k in now if old.get(k) != now[k]]
    if bad:
        print("다르다:", bad, file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)
rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
now.update(git_head=rev, settings={k: v for k, v in sorted(os.environ.items())
                                   if k.startswith(("EXP_", "SIM_", "LLM_", "POLICY_"))})
path.write_text(json.dumps(now, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
