# 서울 ABM 시뮬레이션 — 설치·실행 가이드

이 문서는 **아무것도 모르는 상태**에서 이 레포를 받아 시뮬레이션을 끝까지 돌리는 데
필요한 모든 단계를 적어둔다. 명령어는 그대로 복붙해서 쓸 수 있게 했다.

> 한 줄 요약: **원본 통계 → 에이전트 15,000명 생성 → Neo4j 그래프 구축 → 정책 적재
> → 매일 사이클(아침 계획 → 낮 만족도 → 밤 기억/상호작용) N일 반복 → 결과 분석 + HTML 시각화.**

---

## 0. 시뮬레이션이 무엇을 하는가 (1분 요약)

서울시민 약 1.5만 명을 가상으로 만들어 **하루하루 어디서 무엇을 할지** LLM이 결정하게 한다.

- **인풋**: 행정동 통계(인구·소비·유동인구), 상권 POI 목록, 행정구역 코드, 카테고리 사전, (선택) 정책 문서
- **처리**: 각 에이전트마다 "오늘 어디 갈지" 를 LLM이 계획 → 그래프 DB(Neo4j)에 기록 → 다음 날 그 기억을 참고해 또 계획 → 반복
- **아웃풋**:
  - 일자별 메트릭 JSONL (이벤트 수·만족도·토큰 사용량·정책 카테고리 히트수)
  - 그래프 DB 안에 누적된 `:Plan`, `:Memory`, `:Conversation`, `:State` 노드
  - 모든 결과를 하나의 HTML 파일로 묶은 인터랙티브 지도

**정책 효과는 어떻게 측정하나?** — 임의의 계수(예: "쿠폰이면 만족도 +0.1") 를 쓰지 않는다.
정책의 자연어 설명(`description`, `name`, `benefit_rate`, `cap_per_agent`) 이 매일 아침
각 에이전트의 LLM 프롬프트에 통째로 들어가고, LLM이 그걸 읽어 "오늘 거기 갈까/뭐 살까" 를
스스로 판단한다. 정책의 **영향 크기**는 시뮬 끝나고 *결과* 분포(정책 적용 동 vs 비적용 동
이벤트 비율 차이, 만족도 차이) 에서 사후 측정한다. → 영향도가 미리 박혀있지 않으니
"이 정책이 진짜 효과 있는지" 가 시뮬 출력으로부터 나온다.

---

## 1. 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Windows 11 (WSL2) / Linux / macOS | Linux (Ubuntu 22.04+) |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB+ |
| GPU (LLM 서버) | RTX 5090 32GB (Qwen3-14B-AWQ) | A100 80GB (Qwen3-32B-AWQ) |
| CUDA | 12.1+ | 12.4+ |
| 디스크 | 50 GB (모델 가중치 + 출력) | 100 GB |
| Neo4j | 5.20+ | 5.x 최신 |

LLM 서버는 GPU 없이는 못 띄운다. GPU 없는 노트북에서 코드만 보거나 그래프만 보려면
시뮬 단계 스킵하고 시각화 HTML만 받으면 된다.

---

## 2. 사전 설치 (한 번만)

### 2-1. Python & Git

```bash
# Ubuntu / WSL2
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# macOS
brew install python@3.11 git
```

### 2-2. Neo4j 5.x 설치 (3가지 중 택1)

**옵션 A — Docker (가장 쉬움)**
```bash
docker run -d --name neo4j-abm \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v $HOME/neo4j_data:/data \
  neo4j:5.23
# 웹 콘솔: http://localhost:7474 (id: neo4j / pw: changeme123)
```

**옵션 B — Neo4j Desktop** (Windows GUI 선호 시): https://neo4j.com/download/

**옵션 C — Neo4j Aura** (클라우드 무료 인스턴스): https://neo4j.com/cloud/aura-free/

설치 후 비밀번호를 메모해둔다 — `.env` 에 적어야 한다.

### 2-3. GPU 드라이버 + CUDA (LLM 서버용)

```bash
# WSL2 / Ubuntu
nvidia-smi          # CUDA 12.x 보이면 OK
# 안 보이면: https://developer.nvidia.com/cuda-downloads
```

---

## 3. 레포 클론 & Python 환경

```bash
git clone https://github.com/wikriwiki/kw_26_final_project.git
cd kw_26_final_project
git checkout feat/policy-pipeline-port      # 최신 브랜치

# 가상환경 (클라이언트용 — LLM 서버는 별도 venv 권장)
python3.11 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 가 설치하는 것: `neo4j`, `openpyxl`, `pyyaml`, `requests`, `pydantic`,
`openai` (SGLang/vLLM 호환 클라이언트), `watchdog` (정책 폴더 감시).

추가로 필요할 수 있는 것:
```bash
pip install python-dotenv pandas pyarrow tqdm        # 통계 분석·전처리 단계
```

---

## 4. 환경 변수 설정

이 시뮬은 **두 곳**에서 환경변수를 읽는다. 둘 다 만들어야 한다.

### 4-1. `data/neo4j_load/.env` (Neo4j 인증 — 필수)

```bash
# 파일 위치: 레포 루트 기준 data/neo4j_load/.env
mkdir -p data/neo4j_load
cat > data/neo4j_load/.env <<'EOF'
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme123
NEO4J_DATABASE=neo4j

# (선택) V-WORLD geocoding API key — 주소→좌표 변환 시
VWORLD_API_KEY=
EOF
```

Aura 클라우드 쓰면 `NEO4J_URI=neo4j+s://xxx.databases.neo4j.io` 형식.

### 4-2. 쉘 환경변수 (시뮬·LLM 관련)

`~/.bashrc` 또는 `~/.zshrc` 끝에 (Windows PowerShell 이면 `$PROFILE`):

```bash
# 시뮬 출력 디렉토리 (체크포인트·메트릭 JSONL)
export SIM_OUTPUT_DIR="$HOME/sim_output"

# 시각화 산출물 디렉토리 (export_visualization.py)
export VIZ_OUT_DIR="$HOME/sim_output/visualization"

# LLM 모델 선택: qwen32b | qwen14b | qwen9b | exaone
export LLM_MODE=qwen14b

# LLM 서버 URL (SGLang 기본 30000, vLLM 호환 8000)
export SGLANG_BASE_URL=http://localhost:30000/v1

# (선택) 정책 파이프라인 LLM 추출 시
# export OPENAI_API_KEY=sk-...                   # OpenAI structured output 경로
# export POLICY_STRUCTURED_MODE=openai           # 강제 모드
```

`source ~/.bashrc` 로 반영.

---

## 5. LLM 서버 띄우기 (별도 venv 권장)

LLM 서버는 GPU 가 있는 머신에서 띄운다. 시뮬 클라이언트와는 별도 환경이 좋다.

```bash
# LLM 서버용 별도 venv
python3.11 -m venv ~/.venv-sglang
source ~/.venv-sglang/bin/activate
pip install --upgrade pip

# SGLang (RadixAttention prefix cache 지원 — 우리 5-레이어 프롬프트와 궁합)
pip install "sglang[all]"

# Qwen3-14B-AWQ 기동 (RTX 5090 32GB 권장 — 가장 빠름)
bash scripts/serve/serve_qwen14b.sh

# 또는 A100 80GB 1장이면 32B
bash scripts/serve/serve_qwen32b.sh
```

기동 확인:
```bash
curl http://localhost:30000/v1/models
# {"data":[{"id":"Qwen/Qwen3-14B-AWQ", ...}]}
```

**모델별 권장**:
- `qwen14b` — RTX 5090 32GB / RTX 4090 24GB, 5초/agent
- `qwen32b` — A100 80GB, 9초/agent, 출력 품질 최고
- `qwen9b` — 디버깅용, 어떤 GPU든 빠름
- `exaone` — 국내 대회 제출용 EXAONE-4.5-33B-FP8

---

## 6. Day 0 — Neo4j 그래프 구축

### 6-1. 원본 데이터 준비

`data/raw/` 안에 7대 원본 CSV(텔레콤·카드·KT 유동인구·집계구 결제 등)가 있어야 한다.
데이터가 없으면 합성 데이터 모드로 진행 가능:
```bash
python preprocess_join.py synthetic
```

### 6-2. 통계 산출 + 에이전트 생성

```bash
python preprocess_join.py original           # data/raw → output/original/
python analyze_stats.py                      # output/stats/*.json (7종)

# 에이전트 약 15,000명 생성 (LLM 호출, ~30분 ~ 2시간 GPU 의존)
python generate_agents.py --max-concurrent 16
# 결과: output/agents/agents_final.json
```

스모크 테스트:
```bash
python generate_agents.py --limit 20 --max-concurrent 4
```

검증:
```bash
python validate_vs_raw.py
```

### 6-3. Neo4j 적재 (9단계 자동)

```bash
# 제약조건 먼저
python scripts/neo4j_load/apply_constraints.py

# 일괄 실행 (01_admin → 02_categories → 03_pois → 04_agents → 05_anchors
#            → 06_social → 07_initial_awareness → 08_initial_state → 99_validate)
python scripts/neo4j_load/run_all.py
```

각 단계가 무엇을 적재하는지:
| 단계 | 적재 |
|------|------|
| 01_admin | 25 자치구 + 행정동 노드, HAS_DONG, ADJACENT_TO (≤1.5km) |
| 02_categories | 12 L1 + ~90 L2 카테고리 |
| 03_pois | 거주·직장·상권 POI + IN_DONG + IN_CATEGORY |
| 04_agents | Agent 노드 + 30+ 페르소나 속성 |
| 05_anchors | LIVES_AT, WORKS_AT |
| 06_social | KNOWS (친구 네트워크, strength + relation) |
| 07_initial_awareness | KNOWS_POI + 초기 Memory |
| 08_initial_state | Day 0 :State (잔액·에너지·mood·fatigue) |
| 99_validate | 무결성 체크 |

소요시간: 약 10–30 분 (Neo4j 위치·디스크 의존).

---

## 7. 정책 적재

정책은 두 가지 경로 중 하나로 들어간다.

### 7-1. 수기 검수 JSON (정형화된 정책 — 권장)

```bash
# 예: P007 서울시민 소상공인 응원 쿠폰
python -m scripts.policy_pipeline.inject_json \
    data/neo4j_load/policies/P007.json \
    --deactivate-others-from 2026-05-01
```

JSON 스키마 예:
```json
{
  "id": "P007",
  "name": "서울시민 소상공인 응원 쿠폰",
  "type": "subsidy",
  "description": "서울 전역 소상공인 가맹점 결제 시 일정 비율 환급, 1인 최대 10만원.",
  "benefit_rate": 1.0,
  "cap_per_agent": 100000,
  "announce_date": "2026-04-25",
  "effective_from": "2026-05-01",
  "effective_until": "2026-06-30",
  "target_districts": ["강남구", "마포구", "..."],
  "benefit_categories": []
}
```

- `target_districts` 빈 리스트 = 서울 전역
- `benefit_categories` 빈 리스트 = "모든 업종" (시뮬에서 카테고리 제한 없음으로 해석)
- `description` 은 **반드시 자연어로 정책 효과를 설명**해야 한다 — LLM이 이 문장을 읽고 행동을 정한다.

### 7-2. 자연어 텍스트 (LLM 추출 — 실험용)

```bash
# Watchdog 워커 가동
python -m scripts.policy_pipeline.watch

# 다른 터미널에서 정책 원문을 inbox에 떨굼
cp my_new_policy.txt data/policies/inbox/
# → LLM이 자동 추출 → 검증 → Scope 분석 → Neo4j 적재
```

자세한 내용: [scripts/policy_pipeline/README.md](scripts/policy_pipeline/README.md)

---

## 8. 시뮬레이션 실행

### 8-1. 스모크 테스트 (강남구 100명 × 1일)

```bash
python scripts/sim/run_simulation.py \
    --start 2026-05-01 --days 1 \
    --gu 11680 --limit 100 --workers 8
```

출력 위치:
- `$SIM_OUTPUT_DIR/checkpoints/done_2026-05-01.json` — 처리 완료된 agent ID
- `$SIM_OUTPUT_DIR/checkpoints/failed_2026-05-01.json` — 실패 케이스 (있다면)
- `$SIM_OUTPUT_DIR/metrics/day_2026-05-01.jsonl` — agent별 메트릭 라인

메트릭 한 줄 예시:
```json
{"aid":"AGT_11680...","status":"ok","elapsed":4.2,"n_events":7,"n_includes":6,
 "avg_sat":0.68,"balance":150000,"tokens_in":820,"tokens_out":240,"policy_hits":2}
```

### 8-2. 풀런 (15,000명 × 60일)

```bash
python scripts/sim/run_simulation.py \
    --start 2026-05-01 --days 60 --workers 16
# 소요: GPU에 따라 ~10시간(qwen14b) ~ ~3일(qwen32b)
```

중단됐다 다시 돌려도 OK — `done_<day>.json` 보고 resume.

### 8-3. KPI 평가

```bash
python scripts/sim/evaluate.py --start 2026-05-01 --days 60
# DID(정책 적용 vs 비적용 지역 차이), 환각률, 만족도 분포 등
```

### 8-4. 시각화 HTML

```bash
python scripts/sim/export_visualization.py --start 2026-05-01 --days 60
python scripts/sim/build_standalone_html.py
# 결과: output/sim/visualization/standalone.html (단일 HTML, 오프라인 OK)
```

브라우저로 열면 Leaflet 지도 + agent 타임라인 + 정책 효과 비교 패널.

---

## 9. 디렉토리 구조 한눈에 보기

```
kw_26_final_project/
├── data/
│   ├── raw/                          # 원본 CSV (텔레콤·카드·KT·집계구)
│   ├── neo4j_load/
│   │   ├── .env                      # ★ Neo4j 인증
│   │   ├── admin/KIKcd_H.xlsx        # 행정구역 코드
│   │   ├── categories/categories.yaml
│   │   ├── pois/소상공인...csv
│   │   └── policies/P007.json
│   └── policies/inbox/               # 자연어 정책 드롭 폴더
├── output/
│   ├── original/                     # 전처리 결과
│   ├── stats/*.json                  # 통계 7종
│   └── agents/agents_final.json      # 약 15,000 agent
├── scripts/
│   ├── neo4j_load/                   # Day 0 적재 (01~08 + 99 validate)
│   ├── serve/                        # LLM 서버 기동 sh (qwen14b/32b/9b/exaone)
│   ├── policy_pipeline/              # 정책 추출·검증·적재
│   └── sim/                          # 매일 사이클 시뮬
│       ├── run_simulation.py         # ★ 메인 엔트리
│       ├── dawn_context.py           # 7종 Cypher → 프롬프트 블록
│       ├── stage1_intent.py          # 의도·카테고리 LLM
│       ├── stage2_poi.py             # POI 확정 LLM
│       ├── plan_writer.py            # :Plan 적재 + 만족도 + Night Phase
│       ├── night_interaction.py      # 상호작용 매칭
│       ├── night_intent_llm.py       # 대화 의도 분류
│       ├── evaluate.py               # KPI
│       └── export_visualization.py
├── tests/
└── requirements.txt
```

`SIM_OUTPUT_DIR` (기본 `~/sim_output/`) 은 레포 밖에 두는 게 좋다 (Google Drive
동기화 폴더에 두면 file write 충돌 위험).

---

## 10. 자주 막히는 곳

**Q. `NEO4J_PASSWORD not set`**
→ `data/neo4j_load/.env` 안에 `NEO4J_PASSWORD=...` 있는지 확인.

**Q. `Connection refused` (LLM 호출)**
→ SGLang 서버가 안 떠 있다. `curl http://localhost:30000/v1/models` 로 살아있나 확인.

**Q. agent 생성 후 `agents_final.json` 이 비어있음**
→ LLM이 JSON 스키마 못 맞춤. `generate_agents.py --limit 5 --verbose` 로 raw 출력 봐서
어디서 깨지는지 보고, 더 작은 모델(`qwen9b`)로 디버그.

**Q. 시뮬 도중 `OutOfMemory` (Neo4j)**
→ Neo4j 힙 메모리 부족. Docker면 `-e NEO4J_dbms_memory_heap_max__size=8G` 추가.

**Q. Windows에서 한글 깨짐**
→ 코드 안에서 `sys.stdout.reconfigure(encoding="utf-8")` 호출하지만, PowerShell에서
`chcp 65001` 한 번 쳐주면 확실.

**Q. 정책을 새로 추가하고 싶음**
→ `data/neo4j_load/policies/P008.json` 만들고 `python -m scripts.policy_pipeline.inject_json
data/neo4j_load/policies/P008.json`. 시뮬 다음 회차부터 자동 반영.

**Q. 정책이 시뮬에 반영 안 되는 것 같음**
→ Neo4j 콘솔에서 `MATCH (p:Policy)-[:applied_to]->(t) RETURN p.id, p.name, t.name` 으로
적재됐는지 확인. `effective_from <= today <= effective_until` 인지도 확인. 정책의
`description` 이 비어있으면 LLM이 행동을 바꿀 단서가 없다 — 반드시 자연어로 채워야 한다.

---

## 11. 개발자용: 단위 테스트

```bash
python -m pytest tests/unit/policy_pipeline -v
# 48 passed
```

Neo4j 통합 테스트는 실제 인스턴스가 필요해 CI에서 제외.

---

## 12. 정책 효과 측정 방법 (왜 modifier 가 없는지)

이 시뮬은 "정책이 만족도를 X점 올린다" 같은 **임의 계수를 코드에 박지 않는다**.
이유: 정책 효과 크기를 사전에 정한다면, 그 값은 결국 누군가의 추측이고, 추측한 값이
시뮬 결과에 그대로 나오는 동어반복이 된다.

대신 다음 경로로 효과를 측정한다:

1. 매일 아침 `dawn_context.POLICY_CYPHER` 가 그 agent의 거주·직장 동에 적용되는
   활성 정책을 Neo4j에서 가져온다.
2. 정책의 `name`, `description`, `benefit_rate`, `cap_per_agent`, 적용 지역·카테고리가
   **자연어 텍스트**로 Stage 1 LLM 프롬프트에 들어간다.
3. LLM은 그 문장을 읽고 "오늘 어디 갈지·뭐 살지" 를 페르소나·기억·정책을 종합해 결정.
4. 결정 결과(이벤트 분포·POI 선택·예상 지출)가 그래프에 적재.
5. 시뮬 끝나고 `evaluate.py` 가 **정책 적용 지역 vs 비적용 지역** (또는 **정책 시행 전 vs 후**)
   을 비교 (DID, propensity matching 등) → "정책 효과 크기" 가 출력으로 나옴.

따라서 정책 텍스트의 품질(특히 `description`)이 시뮬 신뢰성을 좌우한다.

---

## 13. 라이선스 / 데이터 출처

- 원본 통계: 서울시 빅데이터캠퍼스, KT 유동인구, 카드사 소비 데이터 등 (각 라이선스 준수)
- 코드: 본 레포 라이선스 참조
- LLM 모델: Qwen3 (Apache 2.0), EXAONE (LG 제공 라이선스)

질문은 GitHub Issue 로.
