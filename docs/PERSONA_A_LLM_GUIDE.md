# 방식 A+LLM 실행 가이드 — rank-coupling 후 모순만 LLM 봉합

방식 A(rank-coupling)로 싸게 짝지은 뒤, **모순이 탐지된 페르소나에만** LLM을 호출해
서사를 숫자에 맞게 재서술하는 경로의 **실행법**을 정리한다.

> 핵심 원칙: 소비·행동 **숫자는 BDC 통계로 확정 = 절대 불변**. LLM은 **서사(lifestyle)만**
> 숫자와 모순 없게 다시 쓴다. 원본 NVIDIA 서사는 보존된다.
> (방법론 비교는 `docs/PERSONA_NVIDIA_METHODS.md` 참고.)

---

## 1. 동작 원리 (한눈)

```
NVIDIA+BDC로 방식 A 생성 (숫자 확정, 서사 부착)
        │
        ▼
규칙으로 모순 사전탐지   ── 모순 없음 ──▶ 그대로 통과 (LLM 호출 안 함)
        │ 모순 있음
        ▼
LLM이 서사 재서술(봉합)  →  숫자 불변, lifestyle만 정합화 + 메타 기록
```

- **LLM은 모순난 것에만 발동** → 전체의 약 10~30%만 호출(비용 절감).
- 서버 없이 검증할 수 있는 **오프라인 stub** 모드 제공.

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

## 6. 모순 탐지 규칙 (LLM 호출 게이트)

`llm_reconcile.detect_contradictions` 가 아래를 검사해 하나라도 걸리면 LLM 호출:

| 규칙 | 조건 | 의미 |
|------|------|------|
| `ses_consume_gap` | \|소비분위(정규화) − SES\| > 0.4 | 고소비↔저SES 또는 저소비↔고SES |
| `luxury_hobby_low_spend` | 고급 취미(와인·골프·명품…) + 소비분위 ≤ 2 | 서사↔소비 모순 |
| `frugal_hobby_high_spend` | 검소 취미 2개↑ + 소비분위 ≥ 9 | 서사↔소비 모순 |
| `high_ses_job_low_spend` | SES ≥ 0.8 전문직 + 분위 ≤ 2 (구직/전직/은퇴 제외) | 직업↔소비 모순 |
| `location_conflict` | 서사가 거주지와 **다른 자치구** 언급 | A 폴백 매칭 부작용 |

임계값 `gap_threshold=0.4`는 휴리스틱(코드에서 조정 가능).

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
    "llm_contradictions": ["ses_consume_gap:...", "location_conflict:..."],
    "llm_reconciled": true,        // 봉합 수행 여부
    "llm_resolution": "무엇을 어떻게 정합화했는지 1문장"
  }
}
```
모순이 없던 페르소나는 `llm_reconciled: false`, `llm_contradictions: []`로 기록되고
LLM 호출은 일어나지 않는다.

실행 후 콘솔 요약:
```
[rank-coupling+LLM] 15000 personas → ...
  match levels: {'gu_sex_age': 5170, 'sex_age': 9830}
  llm(qwen32b): 모순탐지 N/15000, 봉합 M
```

---

## 8. 비용·규모

- **호출 수 = 모순 탐지 건수**(전체 아님). 보통 전체의 10~30%.
- SGLang RadixAttention + 배치 처리로 동시성 높임(자세히 `docs/SGLANG_MIGRATION.md`).
- 모델 선택 가이드: 디버깅 `qwen9b/14b`, 본런 `qwen32b`, 국내대회 `exaone`.

---

## 9. 트러블슈팅

| 증상 | 원인/조치 |
|------|-----------|
| `봉합 0` 인데 모순은 많음 | LLM 응답 JSON 파싱 실패 → `--llm-mode` 변경, 서버 로그 확인. `parse_fix`가 빈 dict면 `llm_reconciled:false` |
| 서버 연결 실패 | `python scripts/sim/llm_client.py` healthcheck로 base_url/served_match 확인. 포트/모델명 불일치 점검 |
| `openai` ImportError | `pip install -r requirements.txt` |
| 데이터 120건만 | fixture 사용 중 → `prepare_nvidia.py`로 full 받기 |
| 메모리 부족(전체) | `--jsonl` 사용 |
| 결과 비결정적 | LLM 생성은 본질적으로 비결정적(temp 0.4). 재현 필요 시 stub 또는 캐싱 레이어 도입 |

---

## 10. 내부 동작 (함수 맵)

| 파일·함수 | 역할 |
|-----------|------|
| `build_rank_coupling.build(..., llm_reconcile=True)` | A 생성 후 각 페르소나에 봉합 적용 |
| `llm_reconcile.detect_contradictions` | 규칙 기반 모순 사전탐지(호출 게이트) |
| `llm_reconcile.llm_reconcile_persona` | 봉합 본체. 숫자 불변, lifestyle/메타 갱신 |
| `llm_reconcile.build_reconcile_prompt` | 프롬프트 빌더(LLM-입력 필드만, `nvidia_reserved` 제외) |
| `llm_reconcile.make_llm_fixer(mode)` | 실서버 fixer (`scripts/sim/llm_client.call_chat`) |
| `llm_reconcile.stub_fixer` | 오프라인 결정적 fixer |
| `llm_reconcile.parse_fix` | LLM 출력에서 JSON 추출(견고) |

> 단위테스트: `tests/unit/persona/test_llm_reconcile.py` (mock fixer로 서버 없이 검증).
