# 페르소나 샘플 (NVIDIA 정성 + BDC 통계 결합)

NVIDIA Nemotron-Personas-Korea(정성 서사)와 우리 BDC 통계(정량 소비·이동)를
결합하는 **3가지 방식**의 출력 샘플. 각 파일은 동일 시드(42)로 생성한 10명 예시.

> LLM 호출 없음. 전부 결정적(seed=42) 통계 샘플링. 전체 재현 시 NVIDIA 서울
> 서브셋을 받아야 매칭률이 올라감 — `data/personas/README.md` 참고.

| 파일 | 방식 | 생성 스크립트 | 핵심 |
|------|------|----------------|------|
| `A_rank_coupling.json` | A · rank-coupling | `build_rank_coupling.py` | SES 순위로 NVIDIA↔통계 짝짓기 |
| `A_rank_coupling_llm.json` | A+LLM · 전수 검증 | `build_rank_coupling.py --llm-reconcile` | A 후 **모든 페르소나를 LLM이 검증**, 모순이면 서사 봉합 (숫자 불변) |
| `B_conditional_graft.json` | B · conditional-graft | `build_conditional.py` | NVIDIA 사람에게 통계를 조건부 부여 |
| `C_hybrid.json` | C · hybrid | `build_conditional.py --reconcile` | B + 규칙기반 모순 검출·봉합 |

> ⚠️ `A_rank_coupling_llm.json` 은 **오프라인 stub**(`--llm-stub`)으로 생성한 자리표시자
> 예시입니다(`_match.llm_resolution` 에 `[STUB]` 표기). 실제 LLM 봉합 결과를 보려면
> SGLang 서버를 띄우고 `--llm-stub` 없이 재생성하세요.

## 재생성

```bash
# A — rank-coupling
python scripts/persona/build_rank_coupling.py --limit 10
# A+LLM — 모든 페르소나 LLM 전수 검증 (서버 필요). 오프라인 검증은 --llm-stub
python scripts/persona/build_rank_coupling.py --limit 10 --llm-reconcile            # 실 서버
python scripts/persona/build_rank_coupling.py --limit 10 --llm-reconcile --llm-stub # 오프라인
# B — conditional-graft
python scripts/persona/build_conditional.py --limit 10
# C — hybrid (B + reconcile)
python scripts/persona/build_conditional.py --limit 10 --reconcile

# 전체(약 15,000명) 생성: --limit 생략
```

## 각 페르소나 공통 구조

```text
agent_id, residence(dong/gu), personal(job/age/gender/income/life_stage),
workplace(미정=null), spending(소비분위·금액·top_categories),
behavior(배달·쇼핑·이동·재택시간), personality(소비성향·lifestyle),
nvidia_persona(LLM 입력용: summary/hobbies/문화배경/혼인/주거/가족/학력),
nvidia_reserved(저장 전용: 직업관·커리어목표 등 — LLM 입력 X),
_match(방식별 추적 메타)
```

`_match` 메타가 방식을 구분:

- **A**: `method=rank-coupling`, `match_level`, `consume_percentile`, `nvidia_ses`
- **B**: `method=conditional-graft`, `dong_pick_level`, `ses_hint`, `hobby_adjust`
- **C**: B와 동일 + `reconciled`(봉합 여부), `warnings`(잔여 모순 감사 로그)

방식별 상세 비교·장단점은 `docs/archive/PERSONA_NVIDIA_METHODS.md` 참고.
