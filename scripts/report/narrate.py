"""narrate.py — 이미 계산된 숫자를 한국어 보고서 문장으로 바꾸는 좁은 LLM 호출.

여기서 LLM은 "어떤 분석을 쓸지, DID가 타당한지" 같은 판단을 하지 않는다 —
그 판단은 catalog.py의 코드가 이미 끝냈다. LLM의 역할은 주어진 JSON 숫자를
문장으로 풀어쓰는 것뿐이다. 이렇게 역할을 좁혀 두면 국내 소형 모델(K-exaone 등,
scripts/sim/llm_client.py의 MODELS에 OpenAI 호환 엔드포인트로 등록해 교체)로
바꿔도 숫자 왜곡·환각 위험이 크지 않다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

from llm_client import call_chat  # noqa: E402

SYSTEM_PROMPT = """당신은 정책효과 시뮬레이션 보고서의 해설 작성자입니다.
이미 계산이 끝난 분석 결과(JSON)가 주어집니다.

규칙:
- JSON에 없는 숫자를 새로 만들거나 재계산하지 마세요. 있는 숫자만 문장으로 풀어쓰세요.
- 통계적 인과를 단정하지 마세요. 표본 수(n)가 작으면 "제한적으로 해석해야 한다"처럼 명시하세요.
- 정책 담당 공무원이 읽는다고 가정하고, 3~5문장의 자연스러운 한국어로 해설을 작성하세요.
- 수치는 JSON 표기를 그대로 인용하세요 (단위·반올림 임의 변경 금지).
- 결과가 기대와 다르거나 효과가 미미해도 그대로 서술하세요 (긍정적으로 포장 금지).
"""


def narrate(label: str, data: dict, ctx: dict, mode: str | None = None) -> str:
    user_prompt = (
        f"[정책] {ctx.get('name', ctx.get('id'))} ({ctx.get('type')})\n"
        f"[분석 항목] {label}\n"
        f"[계산된 데이터]\n{json.dumps(data, ensure_ascii=False, indent=2, default=str)}\n\n"
        "위 데이터만 근거로 해설 문단을 작성하세요."
    )
    try:
        resp = call_chat(mode, SYSTEM_PROMPT, user_prompt, temperature=0.4, max_tokens=500)
        return resp.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        return f"(자동 해설 생성 실패: {e} — 아래 표·차트를 직접 확인하세요.)"
