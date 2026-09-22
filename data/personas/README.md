# 페르소나 외부 데이터

## `nvidia_seoul_sample.json`

NVIDIA **Nemotron-Personas-Korea** 데이터셋에서 `province == "서울"` 인 레코드
120개를 streaming 으로 추출한 샘플.

- 출처: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea
- 라이선스: **CC BY 4.0** (상업·비상업 사용 가능, NVIDIA attribution 필요)
- 용도: 빈약한 페르소나 정성 레이어 보강 (직업관·취미·가치관·가족 서사)

### 필드 (26)
정성 서사 7종(`professional/sports/arts/travel/culinary/family_persona` + `persona`),
구조화(`cultural_background`, `skills_and_expertise(_list)`, `hobbies_and_interests(_list)`,
`career_goals_and_ambitions`), 인구통계(`sex`, `age`, `marital_status`, `military_status`,
`family_type`, `housing_type`, `education_level`, `bachelors_field`, `occupation`,
`district`, `province`, `country`), `uuid`.

## 실데이터로 전체 생성 (production)

레포 동봉 fixture는 120건뿐이라 B/C는 120명까지만 생성된다. 실제 풀런에서는
1M 전체에서 서울 subset(약 13만)을 받아 쓴다. **준비 스크립트**로 한 번에 처리:

```bash
# 0) 의존성 (다운로드 시에만 필요 — 생성 자체는 불필요)
pip install "datasets>=2.14"

# 1) 서울 실데이터 다운로드 + 필터 + 저장 (대용량이라 jsonl 권장)
python scripts/persona/prepare_nvidia.py --jsonl
#   → data/personas/nvidia_seoul_full.jsonl (약 13만건)
#   옵션: --province 경기 / --all(전국 1M) / --limit 2000(스모크)

# 2) 전체 페르소나 생성 (로더가 full 파일을 자동 우선 사용)
python scripts/persona/build_conditional.py --jsonl              # B (~13만명)
python scripts/persona/build_conditional.py --reconcile --jsonl  # C
python scripts/persona/build_rank_coupling.py --jsonl            # A (15,000명, NVIDIA 풀 13만에서 매칭)
python scripts/persona/build_rank_coupling.py --llm-reconcile --jsonl   # A+LLM (SGLang 서버 필요)
```

### 로더 우선순위 (`_common.load_nvidia_seoul`)
1. env `NVIDIA_PERSONA_PATH` (경로 직접 지정) — `.json`/`.jsonl` 모두 가능
2. `nvidia_seoul_full.jsonl`
3. `nvidia_seoul_full.json`
4. `nvidia_seoul_sample.json` (fixture fallback)

> `nvidia_seoul_full.*` 는 용량이 커서 git 커밋 대상이 아니다(.gitignore 권장).
> 각 환경에서 `prepare_nvidia.py` 로 받아 쓴다.
