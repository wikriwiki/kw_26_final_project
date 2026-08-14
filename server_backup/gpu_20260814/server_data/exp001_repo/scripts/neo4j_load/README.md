# Neo4j Day 0 적재 — 실행 가이드

팀원이 이 그래프를 똑같이 재현할 때 보는 진입점 문서.

## TL;DR

```bash
# 1. Neo4j 5.x 서버 실행 중 + .env 채움
cp data/neo4j_load/.env.example data/neo4j_load/.env  # 그리고 비밀번호 채움

# 2. Python 의존성
pip install -r requirements.txt

# 3. 입력 데이터 확보 (data/neo4j_load/ 안에 배치)
#    - admin/KIKcd_H.xlsx
#    - admin/adm_code_mapping.csv
#    - categories/categories.yaml
#    - mapping/mapping_upjong_to_sub.json
#    - pois/소상공인...서울_*.csv
#    - pois/residence.csv
#    - pois/workplace.csv
#    - agents/agents_final.json

# 4. 제약·인덱스 적용
python scripts/neo4j_load/apply_constraints.py
# 또는: cypher-shell -u neo4j -p <pwd> < scripts/neo4j_load/00_constraints.cypher

# 5. 적재 (~5–15분 소요)
python scripts/neo4j_load/run_all.py
```

마지막에 `99_validate`가 노드/엣지 카운트 + 무결성 검사 JSON을 출력함.

---

## 온톨로지 정의 (이거 먼저 읽어야 함)

| 문서 | 범위 |
|---|---|
| [`docs/schedule_generation_plan/agent_ontology.md`](../../docs/schedule_generation_plan/agent_ontology.md) | 정적 노드 5종 (`Agent`/`POI`/`District`/`Dong`/`Category`) + 엣지 |
| [`docs/schedule_generation_plan/runtime_ontology.md`](../../docs/schedule_generation_plan/runtime_ontology.md) | 런타임 노드 (`State`/`Plan`/`Memory`/`Conversation`/`Policy`) + 엣지 |
| [`docs/guides/NEO4J_SETUP_GUIDE.md`](../../docs/guides/NEO4J_SETUP_GUIDE.md) | **한 페이지 통합 가이드 (노션 공유용)** — 환경·DDL·적재 절차 1파일 |
| [`data/neo4j_load/README.md`](../../data/neo4j_load/README.md) | 입력 데이터 명세 + 현황 |

## 시뮬 본체 (Day 0 적재 이후)

Day 0 그래프가 준비되면 다음 단계 = 매일 시뮬:

| 폴더/문서 | 역할 |
|---|---|
| [`scripts/sim/`](../sim/) | Dawn(Stage 1·2) + Plan + Night Phase 1·2·3 통합 메인 루프 |
| [`scripts/serve/`](../serve/) | SGLang 서버 launch (qwen32b / qwen9b / exaone 3종) |
| [`docs/SIM_PILOT_RESULTS.md`](../../docs/SIM_PILOT_RESULTS.md) | 14,560 agent × 3일 풀런 KPI + 정책 DID 분석 |
| [`docs/design/NIGHT_INTERACTION_REPORT.md`](../../docs/design/NIGHT_INTERACTION_REPORT.md) | Night Phase 2 (상호작용·의도 분류) 통합 보고 |
| [`docs/design/NIGHT_NOTION_DIAGRAM_MAPPING.md`](../../docs/design/NIGHT_NOTION_DIAGRAM_MAPPING.md) | 노션 다이어그램 12박스 → 코드 1:1 매핑 검증 |
| [`docs/archive/SGLANG_MIGRATION.md`](../../docs/archive/SGLANG_MIGRATION.md) | vLLM → SGLang 마이그레이션 (RadixAttention·structured output) |

---

## 환경 요구사항

| 항목 | 최소 | 권장 |
|---|---|---|
| OS | Ubuntu 22.04+, Windows + WSL2 Ubuntu | 동일 |
| Neo4j | 5.x Community (실측 5.26.25) | 동일 |
| Java | OpenJDK 21 | 동일 |
| RAM | 8 GB | 12 GB |
| 디스크 여유 | 10 GB | 20 GB |
| CPU | 4코어 | 8코어 |
| Python | 3.10+ | 3.12+ |

Neo4j heap·pagecache 설정 (`/etc/neo4j/neo4j.conf`):
```
server.memory.heap.initial_size=4g
server.memory.heap.max_size=6g
server.memory.pagecache.size=8g
server.default_listen_address=0.0.0.0
```

WSL2 사용 시 Windows에서 접근하려면 둘 중 하나:

**A. portproxy (즉시)** — WSL 재부팅 시마다 IP 갱신 필요
```powershell
$wslIp = (wsl -d Ubuntu -- hostname -I).Trim().Split(' ')[0]
netsh interface portproxy add v4tov4 listenport=7687 listenaddress=127.0.0.1 connectport=7687 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=7474 listenaddress=127.0.0.1 connectport=7474 connectaddress=$wslIp
```

**B. mirrored networking (영구)** — `%UserProfile%\.wslconfig` 작성 후 `wsl --shutdown`
```ini
[wsl2]
networkingMode=mirrored

[experimental]
autoMemoryReclaim=gradual
```

---

## 적재 스크립트 (순서 = run_all.py)

| # | 스크립트 | 입력 | 출력 노드/엣지 |
|---|---|---|---|
| 00 | `00_constraints.cypher` | — | UNIQUE 제약 10종 + 인덱스 18종 |
| 01 | `01_admin.py` | `admin/KIKcd_H.xlsx` + 소상공인 CSV (중심좌표용) | `:District` × 25, `:Dong` × 427, `[:HAS_DONG]`, `[:ADJACENT_TO]` (≤ 1.5km) |
| 02 | `02_categories.py` | `categories/categories.yaml` | `:Category` × 93 (L2 단위, `parent`로 L1 보존) |
| 03 | `03_pois.py` | `pois/residence.csv`, `pois/workplace.csv`, 소상공인 CSV, `mapping/mapping_upjong_to_sub.json` | `:POI` × ~58만 + `[:IN_DONG]` + `[:IN_CATEGORY]` (commerce만) |
| 04 | `04_agents.py` | `agents/agents_final.json` | `:Agent` × N (페르소나 flat 복제 포함) |
| 05 | `05_anchors.py` | (그래프 사용) | `[:LIVES_AT]`, `[:WORKS_AT {commute_min}]`. 미사용 residence/workplace POI cleanup |
| 06 | `06_social.py` | (그래프 사용) | `[:KNOWS {strength, relation}]` 양방향. 같은 work_dong·home_dong 그루핑 |
| 07 | `07_initial_awareness.py` | (그래프 사용) | `[:KNOWS_POI {source:'initial', since, affinity:0.5}]` — 거주 동 Top-40 + 직장 동 Top-30 + 랜드마크 10. **Memory 노드는 만들지 않음** |
| 08 | `08_initial_state.py` | (그래프 사용) | `:State` × N + `[:HAS_STATE]` — Day 0 시드 (balance, energy, mood, fatigue) |
| 99 | `99_validate.py` | (그래프 사용) | JSON 출력 — 노드/엣지 카운트 + LIVES_AT 누락 + works_at_misassigned 등 |

**선택 스크립트** (run_all에는 없음):
- `03a_residence_from_kapt.py` — K-apt 단지정보 → residence.csv geocoding (V-WORLD API 필요)
- `03b_workplace_from_bldg.py` — 건축물대장 → workplace.csv geocoding

---

## 예상 적재 결과 (14,881 agent 기준)

### 적재 직후 (cleanup 전)

| 항목 | 카운트 |
|---|---|
| District / Dong / Category | 25 / 427 / 93 |
| POI (residence / workplace / commerce) | 3,146 / 38,438 / 537,489 |

### `05_anchors.py` cleanup 후 (실측 — 미사용 POI 자동 제거)

| 항목 | 카운트 |
|---|---|
| District / Dong / Category | 25 / 427 / 93 |
| POI | **543,924** (residence 2,909 + workplace 3,526 + commerce 537,489) |
| Agent / State | 14,881 / 14,881 |
| **노드 합** | **~559K** |
| HAS_DONG / IN_DONG / IN_CATEGORY / ADJACENT_TO | 427 / 543,924 / 537,489 / 2,642 |
| LIVES_AT / WORKS_AT | 14,560 / 8,876 |
| KNOWS / KNOWS_POI | 159,914 / 917,564 |
| HAS_STATE | 14,881 |
| **엣지 합** | **~2.2M** |

> `cleanup`은 `05_anchors.py` 끝에서 `LIVES_AT`/`WORKS_AT`이 연결되지 않은 residence/workplace POI를 DETACH DELETE — workplace 92.8% 감축 (38,438 → 3,526). commerce는 대상 아님.

디스크 사용: 데이터베이스 본체 **~947MB** + 트랜잭션 로그 ~3GB (시뮬 진행 후 +2~5GB).

---

## 알려진 데이터 한계

- LIVES_AT 누락 ~2% — `agents_final.json`의 일부 `residence.dong_code`가 KIK 폐지 동
- WORKS_AT nearest fallback ~19% — workplace POI 풀이 V-WORLD geocode 한도로 26%만 채워짐 (`03b_workplace_from_bldg.py` 재실행 필요)
- workplace POI name 47% 비어 있음 (동일 사유)

---

## 동작 검증 쿼리 예시

Day 0 적재가 잘 됐는지 Cypher로 확인:

```cypher
// Dawn ⑦ Stage 2 candidate (한 agent의 직장 동 한식 POI)
MATCH (a:Agent {id:$aid})-[:WORKS_AT]->(:POI)-[:IN_DONG]->(wd:Dong)
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(wd)
MATCH (p)-[:IN_CATEGORY]->(c:Category {name:'한식'})
OPTIONAL MATCH (a)-[kp:KNOWS_POI]->(p)
RETURN p.name, kp.affinity, (kp IS NOT NULL) AS known
ORDER BY known DESC LIMIT 10;

// 라벨별 카운트
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;
```

---

## 다음 — 시뮬 본체 실행

Day 0 적재 후 매일 시뮬:

```bash
# SGLang 서버 (Qwen3-32B-AWQ)
bash scripts/serve/serve_qwen32b.sh

# 시뮬 (3일치, 강남 100명 예시)
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --gu 11680 --limit 100 --workers 16

# 풀런 (14,560 agent × 3일, ~22시간 — SGLang은 ~13-15h 예상)
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --workers 16

# KPI 평가
python scripts/sim/evaluate.py --start 2026-05-01 --days 3
```

환경변수 (전체 시뮬 공통):
- `SIM_OUTPUT_DIR` (기본 `~/sim_output`) — 체크포인트·메트릭 저장
- `VIZ_OUT_DIR` (기본 `<프로젝트루트>/output/sim/visualization`) — 시각화 JSON
- `LLM_MODE` (기본 `qwen32b`) — `qwen9b` (개발용), `exaone` (국내 대회용)
- `SGLANG_BASE_URL` (auto-detect: 30000 → 8000) — LLM 서버 URL

자세한 내용은 위 시뮬 본체 섹션의 문서 링크 참조.
