# 서울 상권정책 시뮬레이션

소상공인 상권정책의 효과를 LLM 기반 에이전트 시뮬레이션으로 평가하는 프로젝트.

서울 시민 약 14,881명의 페르소나를 빅데이터캠퍼스(BDC) 통계로 생성하고, Neo4j 그래프 위에서 일별 의도·POI 선택·상호작용·정책 수용을 LLM(Qwen3) 으로 시뮬레이션. 정책 시행 전/후의 강남 매출과 비강남 대조군 변화를 Difference-in-Differences (DID) 로 측정한다.

자세한 셋업·실행 절차는 [SETUP.md](SETUP.md) 참고.

---

## 디렉토리 구조

```
final_project/
├─ scripts/             # 모든 실행 가능한 코드
│  ├─ bdc/              # 빅데이터캠퍼스 전처리·검증·페르소나 생성 파이프라인
│  ├─ neo4j_load/       # Neo4j 그래프 초기 적재 (Day 0)
│  ├─ sim/              # 일별 시뮬 (Dawn → Stage1/2 → Plan → Night)
│  ├─ policy_pipeline/  # 정책 JSON 인제스션 + 스케줄링
│  ├─ geocode/          # POI 좌표 보강 (VWorld API)
│  └─ serve/            # vLLM / SGLang 서버 부팅 스크립트
├─ data/                # 원본·중간 데이터 (대부분 .gitignore)
│  ├─ raw/              # BDC 원본 7대 CSV/ZIP (외부 반입)
│  ├─ mapping/          # 코드 매핑 (mopas_nso 등)
│  ├─ policies/         # 정책 JSON (raw → processed → failed)
│  └─ neo4j_load/       # 적재용 중간 파일 (POI, agent)
├─ docs/                # 설계 문서·런북
│  ├─ BDC/              # BDC 데이터 핸들링·전처리 결정
│  └─ schedule_generation_plan/  # 스케줄 생성·온톨로지 설계 문서
├─ output/              # 생성물 (대부분 .gitignore)
│  ├─ stats/            # 페르소나 생성용 통계 JSON 10종 (트래킹)
│  ├─ agents/           # agents_final.json
│  ├─ sim/              # 시뮬 메트릭·시각화 dump
│  │  └─ report/        # 최종 보고서 (FINAL_REPORT_*.{md,html} + 차트 PNG)
│  └─ policy_pipeline/  # 정책 처리 결과
├─ tests/               # pytest 단위 테스트
├─ prototype/           # 옛 시뮬 잔재 (별도 브랜치, gitignore)
├─ logs/                # 런타임 로그 (gitignore)
├─ SETUP.md             # 셋업·실행 가이드
├─ requirements.txt     # 파이썬 의존성
├─ conftest.py          # pytest 루트 설정 (sys.path 보정)
├─ .gitignore           # 추적 제외 패턴
└─ agents.zip           # 트래킹된 에이전트 데이터 아티팩트
```

---

## scripts/ 세부

### scripts/bdc/ — 빅데이터캠퍼스 전처리·페르소나 생성

상세 파이프라인 문서: [scripts/bdc/README.md](scripts/bdc/README.md)

| 파일 | 역할 |
|---|---|
| `file_discovery.py`         | data/raw/ 안의 압축·CSV 자동 탐지 (FileEntry 추상화) |
| `preprocess_join.py`        | 원본 CSV → 조인된 중간 테이블 (output/original/, output/synthetic/) |
| `analyze_stats.py`          | 조인 결과 → 페르소나 생성용 통계 JSON (output/stats/*.json, 7종) |
| `generate_agents.py`        | 통계 JSON + LLM → 약 15,000명 페르소나 (output/agents/agents_final.json) |
| `validate_vs_raw.py`        | 생성된 페르소나 분포 vs 원본 분포 검증 (트래킹된 4개 엔트리포인트 중 하나) |
| `patch_failed_joins.py`     | preprocess_join에서 실패한 행 사후 보정 |
| `synthetic_generator.py`    | data/synthetic/ 합성 데이터 생성 (BDC 미반입 시 fallback) |
| `compare_stats.py`, `compare_to_json.py` | 통계 비교·검증 |
| `validate_agents.py`, `validate_pandas_*.py` | 페르소나·전처리 결과 검증 |
| 기타 보조: `assign_income_bucket.py`, `extract_ksco_codes.py`, `match_occupation_to_ksco.py`, `inspect_personas.py`, `count_seoul.py`, `quick_validate.py` |

모든 파일 헤더에 `PROJECT_ROOT = Path(__file__).resolve().parents[2]` 로 프로젝트 루트 앵커. 어느 디렉토리에서 호출해도 `data/`, `output/` 경로가 정확히 해석된다.

### scripts/sim/ — 일별 시뮬레이션

**2-Stage LLM 파이프라인**
| 파일 | 역할 |
|---|---|
| `dawn_context.py`        | Dawn — Neo4j에서 페르소나·State·Memory·Conversation·약속 fetch + 프롬프트 블록 |
| `stage1_intent.py`       | Stage 1 — LLM이 의도·카테고리·anchor 시퀀스 결정. trigger normalize 헬퍼 포함 |
| `stage2_poi.py`          | Stage 2 — Stage 1 이벤트별 POI 선택. desire-curve 기반 후보 정렬 + 풀 분할 |
| `desire.py`              | POI 방문 욕구 함수 (baseline × recency × saturation + novelty), pure |
| `visit_window.py`        | KNOWS_POI 30일 슬라이딩 윈도우 Python 헬퍼 |
| `plan_writer.py`         | Stage 1+2 → Plan/INCLUDES 노드/엣지 작성 + 만족도 룰 적용 |

**Night Phase**
| 파일 | 역할 |
|---|---|
| `night_interaction.py`   | 상호작용 쌍 선정 — 3축 점수(노출·관계·긴급) + Softmax 확률 매칭 + 친밀도 시간 감쇠 |
| `night_intent_llm.py`    | 상호작용 의도 분류 LLM (약속/추천/이슈/기타) |
| `backfill_night_reasoning.py` | Night reasoning 사후 보완 |

**오케스트레이션·분석**
| 파일 | 역할 |
|---|---|
| `run_simulation.py`      | 일별 시뮬 메인 루프 (workers 병렬, agent별 process_one) |
| `llm_client.py`          | OpenAI 호환 client (singleton thread-safe). vLLM / SGLang auto-detect |
| `evaluate.py`            | 결과 평가 |
| `interview_agent.py`     | 시뮬 종료 후 페르소나별 1대1 인터뷰 (positive/negative/neutral 샘플) |
| `analyze_repeat_visits.py` | 같은 POI 반복 방문 패턴 분석 |
| `day_health_check.py`    | 1일치 풀런 결과 빠른 진단 markdown |
| `aggregate_fallback_stats.py` | fb_* 카운터 집계 |
| `export_visualization.py`, `build_standalone_html.py` | 시각화 JSON dump + 단일 HTML 번들 |
| `generate_final_report.py` | 최종 보고서 (DID, spillover, trigger 분포, 인터뷰 등) |

### scripts/neo4j_load/ — Neo4j 초기 적재 (Day 0)

`00_constraints.cypher` → `01_admin.py` → `02_categories.py` → `03_pois.py` → `04_agents.py` → `05_anchors.py` → `06_social.py` → `07_initial_awareness.py` → `08_initial_state.py` → `99_validate.py` 의 9단계. `run_all.py` 로 일괄 실행.

`backfill_category_desire_params.py` 는 기존 그래프에 desire 파라미터(recovery_tau_days/desire_drop/saturation_n)만 추가하는 멱등 backfill.

`_common.py` 는 driver 싱글톤(thread-safe, connection pool 공유).

### scripts/policy_pipeline/ — 정책 인제스션

정책 JSON 원본 → 추출(extractor) → 검증(validator) → Neo4j 작성(neo4j_writer) → 시뮬 적재 시간 스케줄(schedule.py). 상세는 [scripts/policy_pipeline/README.md](scripts/policy_pipeline/README.md).

### scripts/geocode/ — POI 좌표 보강

VWorld API로 주소 → 위경도. `cache.sqlite` 에 결과 영속화.

### scripts/serve/ — LLM 서버 부팅

| 파일 | 모델 |
|---|---|
| `serve_qwen32b.sh` | Qwen3-32B-AWQ — **현재 표준** (페르소나/스케줄/Graphiti 통합) |
| `serve_qwen14b.sh` | Qwen3-14B-AWQ — 7일 풀런에 사용 |
| `serve_qwen9b.sh`  | Qwen3-9B — 디버그·smoke test |
| `serve_exaone.sh`  | EXAONE 32B — 비교 실험 |
| `run_vllm.sh`      | vLLM 일반 부팅 헬퍼 |

---

## 핵심 실행 흐름

```
[BDC 원본]  →  scripts/bdc/preprocess_join.py  →  scripts/bdc/analyze_stats.py
              ↓                                      ↓
        output/original/                       output/stats/*.json
                                                     ↓
                                  scripts/bdc/generate_agents.py
                                                     ↓
                                  output/agents/agents_final.json
                                                     ↓
                              scripts/neo4j_load/run_all.py  → Neo4j Day 0
                                                     ↓
                              scripts/sim/run_simulation.py  → 일별 시뮬
                                                     ↓
                          scripts/sim/generate_final_report.py
                                                     ↓
                         output/sim/report/FINAL_REPORT_*.{md,html}
```

각 단계별 상세 커맨드는 [SETUP.md](SETUP.md) 참고.

---

## 시뮬레이션 디자인 메모

- **그래프 백엔드**: Neo4j 단일 저장소. 컨텍스트 빌더가 Cypher 사전 조회 → LLM 2-Stage. agentic RAG 아님.
- **정책 수용 모델**: 6단계 라이프사이클(S0~S5) + `policy_baseline` 트레잇으로 조기수용자 약 20% 시딩.
- **상호작용 매칭**: Softmax 확률 선택 (`temperature=0.5` 기본). 친밀도는 마지막 대화 이후 시간 감쇠 (7일 반감).
- **POI 욕구 점수**: `baseline(affinity, sat) × recency(Δ) × saturation(v30) + novelty` 의 곱셈 조합. 어제 단골이라도 desire 가 낮으면 새 가게로 분산.
- **trigger 라벨**: `appointment | rumor | policy | lifestyle | top_category | mood | none` 7종 표준. `habit`/`life_style` 은 모두 `lifestyle` 로 정규화.

---

## 참고 문서

- [SETUP.md](SETUP.md) — 환경 셋업·실행 절차
- [docs/schedule_generation_plan/](docs/schedule_generation_plan/) — 시뮬 설계·온톨로지
- [docs/BDC/](docs/BDC/) — BDC 데이터 핸들링 결정·런북
- [output/sim/report/FINAL_REPORT_7D.md](output/sim/report/FINAL_REPORT_7D.md) — 최신 7일 풀런 결과
- [scripts/bdc/README.md](scripts/bdc/README.md) — Agent Persona Pipeline (전처리→통계→생성→검증)
- [scripts/policy_pipeline/README.md](scripts/policy_pipeline/README.md) — 정책 인제스션 파이프라인
