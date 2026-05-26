# 방식 A+LLM 실행 가이드 — 모든 페르소나를 LLM이 전수 검증·봉합

방식 A(rank-coupling)로 짝지은 뒤, **모든 페르소나를 LLM이 직접 검증**해 한 사람으로
자연스러운지 판단하고, 모순이면 서사를 숫자에 맞게 재서술하는 경로의 **실행법**을 정리한다.

> 핵심 원칙:
> - 소비·행동 **숫자는 BDC 통계로 확정 = 절대 불변**. LLM은 **서사(lifestyle)만** 봉합.
> - **룰 기반 사전탐지를 쓰지 않는다.** 규칙은 부정확하므로, LLM이 전수 판정한다.
> - 프롬프트에는 합격/불합격 **기준(임계값·규칙)을 적지 않고**, LLM이 **무엇을 보고
>   판단할지(정보 차원)** 만 안내한다 → 룰 베이스와 구분.
> - 원본 NVIDIA 서사는 보존된다. (방법론 비교는 `docs/PERSONA_NVIDIA_METHODS.md`.)

---

## 1. 동작 원리 (한눈)

```
NVIDIA+BDC로 방식 A 생성 (숫자 확정, 서사 부착)
        │
        ▼
모든 페르소나를 LLM이 전수 검증 (한 사람으로 자연스러운가?)
        │
   ┌────┴─────────┐
 일관됨          모순 발견
   │                │
 그대로 통과     서사 재서술(봉합) — 숫자 불변, lifestyle만 정합화 + 메타 기록
```

- **모든 페르소나가 LLM 호출 대상** (사전 규칙 게이트 없음). 비용은 §8 참고.
- 서버 없이 배선/테스트할 수 있는 **오프라인 stub** 모드 제공(단순 placeholder).

---

## 2. 사전 준비

### 2-1. 파이썬 의존성
```bash
pip install -r requirements.txt        # openai 포함 (LLM 호출용)
# 실데이터 다운로드까지 하려면:
pip install "datasets>=2.14"
```

### 2-2. NVIDIA 데이터 (선택: 실데이터 풀런)
fixture(120건)로도 동작하지만, 실제로는 서울 풀(~13만)을 받는다.
```bash
python scripts/persona/prepare_nvidia.py --jsonl
#   → data/personas/nvidia_seoul_full.jsonl (로더가 자동 우선 사용)
```

### 2-3. LLM 서버 (실서버 실행 시 필수)
SGLang(권장, 포트 30000) 또는 vLLM(포트 8000) 중 하나를 띄운다. 레포에 기동
스크립트가 있다.
```bash
# 예: 개발/디버깅용 14B (vLLM, 포트 8000)
bash scripts/serve/serve_qwen14b.sh
# 대회/본런: 32B AWQ
bash scripts/serve/serve_qwen32b.sh
# 그 외: serve_qwen9b.sh, serve_exaone.sh
```
서버 연결 확인:
```bash
python scripts/sim/llm_client.py        # healthcheck JSON 출력 (base_url/active_model/served_match)
```
> `llm_client`가 포트를 자동 감지한다: **SGLang(30000) 우선 → 없으면 vLLM(8000) 폴백.**
> 자세한 서버 셋업은 `docs/SGLANG_MIGRATION.md`.

---

## 3. 실행

### 3-1. 오프라인 검증 (서버 불필요) — 먼저 이걸로 파이프라인 확인
```bash
python scripts/persona/build_rank_coupling.py --limit 10 --llm-reconcile --llm-stub \
  --out output/personas/samples/A_rank_coupling_llm.json
```
- `--llm-stub`: 결정적 규칙 기반 자리표시자 fixer 사용. 출력 `_match.llm_resolution`에
  `[STUB]` 표기가 붙는다. 실제 LLM 결과 아님 — **배선 확인·테스트용**.

### 3-2. 실서버 실행 (소량 스모크)
```bash
# (서버가 떠 있는 상태)
python scripts/persona/build_rank_coupling.py --limit 50 --llm-reconcile --llm-mode qwen14b
```

### 3-3. 전체 실행 (실데이터 풀런)
```bash
python scripts/persona/build_rank_coupling.py --llm-reconcile --llm-mode qwen32b --jsonl
#   --jsonl: 대용량 메모리 절약(라인당 1건). --limit 생략 = 전체(15,000명)
```

---

## 4. CLI 플래그 (`build_rank_coupling.py`)

| 플래그 | 기본 | 설명 |
|--------|------|------|
| `--llm-reconcile` | off | **A+LLM 활성화.** 모순 페르소나만 LLM 봉합 |
| `--llm-stub` | off | 서버 없이 결정적 stub fixer (오프라인/테스트) |
| `--llm-mode MODE` | env/기본 | `qwen32b`(기본)·`qwen14b`·`qwen9b`·`exaone` |
| `--jsonl` | off | JSONL 라인 출력(대용량 권장). `.json`→`.jsonl` 자동 |
| `--limit N` | 0(전체) | 생성 수 제한(스모크 테스트) |
| `--seed N` | 42 | 결정성 시드 |
| `--out PATH` | samples/A_rank_coupling.json | 출력 경로 |

> `--llm-reconcile` 없이 실행하면 순수 방식 A(LLM 미사용).

---

## 5. 환경변수 (LLM 서버 연결)

| 변수 | 용도 | 예 |
|------|------|----|
| `LLM_MODE` | 모델 모드(=`--llm-mode` 대체) | `qwen14b` |
| `SGLANG_BASE_URL` | SGLang 서버 URL | `http://localhost:30000/v1` |
| `LLM_BASE_URL` | 대체 서버 URL | `http://localhost:8000/v1` |

우선순위: `--llm-mode` > `LLM_MODE` > 기본 `qwen32b`. URL 미지정 시 30000→8000 자동 감지.

```bash
export LLM_MODE=qwen14b
export SGLANG_BASE_URL=http://localhost:30000/v1
python scripts/persona/build_rank_coupling.py --llm-reconcile --jsonl
```

---

## 6. LLM이 무엇을 보고 판단하나 (기준 아님, 정보 차원)

룰 기반 사전탐지는 부정확해서 **쓰지 않는다.** 대신 모든 페르소나를 LLM에 넘기고,
프롬프트에는 **판정 기준(임계값·규칙)을 적지 않은 채** "무엇을 보고 종합 판단할지"만
안내한다(`llm_reconcile._SYSTEM`). LLM이 보는 정보 차원:

- 소비 수준(분위·하루 지출액·소득 라벨)과 **직업·학력**이 함의하는 경제적 여건
- 소비 수준이 **서사·취미**에서 드러나는 씀씀이·생활양식과 어울리는지
- **거주지(자치구·동)** 와 서사가 언급·암시하는 생활 반경·지역의 일치 여부
- **행동 지표**(배달 빈도·이동 거리·재택 시간·주요 소비 카테고리)와 서사의 라이프스타일
- **생애단계·혼인·가족 구성**이 위 모든 것과 자연스럽게 맞물리는지

> LLM에게 "수치 임계값이나 기계적 규칙으로 판정하지 말고, 한 인간으로서 전체 맥락이
> 말이 되는지 직관적·종합적으로 보라"고 명시한다. 이것이 룰 베이스와의 핵심 차이.
> (모든 기준을 프롬프트에 나열하면 룰 베이스와 같아지므로 의도적으로 적지 않는다.)

---

## 7. 출력 구조 (무엇이 바뀌나)

봉합된 페르소나에서 **숫자는 그대로**, 서사와 메타만 갱신된다.

```jsonc
{
  "spending":  { ... },          // ← 불변 (BDC 통계)
  "behavior":  { ... },          // ← 불변
  "personality": {
    "lifestyle": "봉합된 융합 서술"  // ← LLM이 재서술 (최대 200자)
  },
  "nvidia_persona": {
    "summary": "원본 NVIDIA 서사",   // ← 보존
    "fused_lifestyle": "봉합 서술"   // ← 추가 (다운스트림 LLM 입력용)
  },
  "_match": {
    "method": "rank-coupling",
    "llm_audited": true,           // 전수 검증 — 항상 true
    "llm_consistent": false,       // LLM 판정: 일관/모순
    "llm_reconciled": true,        // 봉합(서사 재서술) 수행 여부
    "llm_issues": "LLM이 본 모순 요약",
    "llm_resolution": "어떻게 정합화했는지 1문장"
  }
}
```
일관 판정된 페르소나는 `llm_consistent: true`, `llm_reconciled: false`로 기록되고
서사·숫자는 그대로 유지된다(전수 검증이라 `llm_audited`는 항상 true).

실행 후 콘솔 요약:
```
[rank-coupling+LLM] 15000 personas → ...
  match levels: {'gu_sex_age': 5170, 'sex_age': 9830}
  llm(qwen32b): 전수검증 15000/15000, 모순발견 N, 봉합 M
```

---

## 8. 비용·규모 (중요)

- **호출 수 = 전체 페르소나 수** (전수 검증). 방식 A는 15,000 → LLM 콜 15,000회.
  룰 게이트 버전(모순난 것만)보다 **호출이 많다** — 정확도(LLM 종합 판단)와의 트레이드오프.
- SGLang RadixAttention + 배치 처리로 동시성을 높여 처리(자세히 `docs/SGLANG_MIGRATION.md`).
  시스템 프롬프트가 전 호출 공통이라 prefix cache 적중률이 높음.
- 비용 절감이 필요하면: 작은 모델(`qwen14b`)로 검증, 또는 `--limit`으로 표본 검증 후 확대.
- 모델 선택 가이드: 디버깅 `qwen9b/14b`, 본런 `qwen32b`, 국내대회 `exaone`.

---

## 9. 트러블슈팅

| 증상 | 원인/조치 |
|------|-----------|
| `모순발견`은 많은데 `봉합 0` | LLM이 `fused_lifestyle`을 비워 보냄/JSON 파싱 실패 → `--llm-mode` 변경, 서버 로그 확인. `parse_verdict`가 빈 dict면 `llm_consistent:true`로 처리 |
| 전부 `llm_consistent:true` | 모델이 너무 관대 → 더 큰 모델(`qwen32b`)로, 또는 프롬프트 강화 |
| 서버 연결 실패 | `python scripts/sim/llm_client.py` healthcheck로 base_url/served_match 확인. 포트/모델명 불일치 점검 |
| `openai` ImportError | `pip install -r requirements.txt` |
| 데이터 120건만 | fixture 사용 중 → `prepare_nvidia.py`로 full 받기 |
| 메모리 부족(전체) | `--jsonl` 사용 |
| 결과 비결정적 | LLM 생성은 본질적으로 비결정적(temp 0.3). 재현 필요 시 stub 또는 캐싱 레이어 도입 |
| 비용 과다 | 전수 호출이라 콜이 많음(§8). 작은 모델 또는 `--limit` 표본 |

---

## 10. 내부 동작 (함수 맵)

| 파일·함수 | 역할 |
|-----------|------|
| `build_rank_coupling.build(..., llm_reconcile=True)` | A 생성 후 각 페르소나를 LLM이 전수 검증 |
| `llm_reconcile.llm_audit_persona` | 감수 본체. judge 호출→일관/모순 기록, 모순이면 서사 봉합(숫자 불변) |
| `llm_reconcile.build_audit_prompt` | 프롬프트 빌더(정보 차원만 안내, `nvidia_reserved` 제외) |
| `llm_reconcile.make_llm_judge(mode)` | 실서버 judge (`scripts/sim/llm_client.call_chat`) |
| `llm_reconcile.stub_judge` | 오프라인 결정적 judge (단순 placeholder) |
| `llm_reconcile.parse_verdict` | LLM 출력에서 JSON 추출(견고) |

> 단위테스트: `tests/unit/persona/test_llm_reconcile.py` (mock fixer로 서버 없이 검증).
