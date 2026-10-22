# 정책 시뮬레이션 웹 콘솔

S1 계약은 [`CONTRACT.md`](CONTRACT.md)와 `fixtures/`에 잠겨 있다. API는
`api.app:app`, React 번들은 `ui/`에 있다.
보고서 엔진은 [`../docs/REPORT_V2.md`](../docs/REPORT_V2.md)에 따로 적었다.

## 로컬 실행

```powershell
# 1) 환경 변수 — 처음 한 번
Copy-Item .env.example .env      # 필요한 값만 채운다

# 2) API (정적 번들까지 8000 포트에서 서빙)
python -m uvicorn web.api.app:app --host 127.0.0.1 --port 8000

# 3) 프런트를 고칠 때만 — 5173 포트, /api 는 8000 으로 프록시된다
cd web/ui
npm install
npm run dev
```

`npm run build` 후에는 FastAPI가 `web/ui/dist/`를 정적으로 서빙한다. 배포
대상 호스트와 SSH/인증 결정은 기준서 지시에 따라 사람 확인 전까지 실행하지
않는다.

## 환경 변수

| 변수 | 뜻 |
|---|---|
| `SIM_DATA_ROOT` | 완료된 run 들이 있는 루트 (`out_BASE` / `out_FINAL` / `rescue/out_BASE7500`) |
| `SIM_USE_FIXTURES=1` | S1 픽스처 provider. **테스트 전용** — 실제 보고서는 만들 수 없다 |
| `SIM_RUN_COMMAND_JSON` | 실행 버튼이 띄울 **고정 명령**. 없으면 `/api/runner/start` 는 503 |
| `SIM_LOCK_PATH` | 실행 lock 파일 (기본 `web/.run.lock`) |
| `WEB_CORS_ORIGINS` | 프런트 개발 서버 주소 |
| `SIM_VIZ_FILE` | 운영자가 직접 걸어 둔 3D 지도 파일. `SIM_VIZ_STANDALONE` → `SIM_DATA_ROOT/viz/sim_demo.html` 순으로 물러난다 |
| `SIM_VIZ_PACKS_DIR` | 3D 지도 데이터 팩 폴더들의 부모(`<팩>/manifest.json`). 없으면 `SIM_DATA_ROOT/viz/packs` → `web/viz_store/packs` 순으로 찾는다. 팩은 `python -m scripts.sim.visualization_3d.pack --src <JSON 4종 폴더> --out <부모>/<팩 이름>` 으로 만든다 |
| `SIM_EXPERIMENTS_ROOT` | 실험 미러 폴더(기본 `~/sim_data/experiments`). 실험 정의는 `web/experiments/*.json`, 받아 온 실행 기록은 `<폴더>/<실험>/arms/<팔>/` |
| `SIM_REMOTE_<PROFILE>` | 실험 정의의 `remote.profile` 이 가리키는 **읽기 전용** 접속 명령 (예: `SIM_REMOTE_NO_SMOKING_VAST="ssh -i ~/.ssh/no_smoking_vast -p 45025 root@<서버>"`). `python -m web.api.experiment_sync --experiment <id>` 또는 콘솔의 "지금 받기"가 쓴다. 서버에서는 `find`·`cat`·`tail -c`·`head -c|sha256sum`·`tar -c` 만 실행한다 |
| `GEMINI_API_KEY` | 보고서 해설 LLM. 없으면 해설만 규칙 기반으로 대체된다. **인터뷰 화면도 같은 키를 쓴다** |
| `INTERVIEW_BASE_URL` / `_MODEL` / `_API_KEY` | 인터뷰 전용 엔드포인트(OpenAI 호환). 셋이 다 있으면 이쪽을 먼저 쓴다. 주소가 이 PC(`127.0.0.1`·`localhost`, 예: SSH 터널)면 키 없이 쓴다. 화면에 보일 이름은 `INTERVIEW_MODEL_LABEL` |

산출물이 없다고 해서 fixture 로 자동 fallback 하지 않는다. 없으면 없다고 답한다.

## 바로 띄우기 — 다 만든 콘솔 데이터 받기

정책 본런 다섯 개(민생회복 소비쿠폰 1차 · 상생소비지원금 · 긴급재난지원금 1차 · 사회적 거리두기 2단계 ·
서울시 재난긴급생활비 3주)를 콘솔에서 바로 볼 수 있게 만들어 둔 데이터가 구글 드라이브에 있다.

- 받기: <https://drive.google.com/drive/folders/1J6PXPQssuHgm8fX5EvNHtSztDkQIaw4R>
  (`KW_시뮬결과_공유_20261011/WEB_CONSOLE_DATA/kw_web_data_20261022.zip`, 0.87GB, 풀면 5.4GB ·
  SHA-256 `f2c8ecaf214994717a665204364f231ee474d3a2402d1282aa170445c7b95676`)
- 담긴 것: `runs/`(정책별 실행) · `experiments/`(정책 있음·없음 비교) · `packs/`(3D 지도) ·
  `policies/`(본런 고정 정책) · `run_console.sh` / `run_console.ps1`
- 키는 들어 있지 않다. 인터뷰·보고서 해설은 `web/.env` 에 K-EXAONE 키를 따로 넣어야 켜진다.

```bash
# 1) zip 을 아무 데나 푼다 → kw_web_data/
# 2) 처음 한 번: UI 빌드
cd 웹_콘솔/web/ui && npm ci && npm run build
# 3) 띄우기 (http://127.0.0.1:8765)
CONSOLE=<저장소>/웹_콘솔 bash <푼 곳>/kw_web_data/run_console.sh
#    PowerShell: <푼 곳>\kw_web_data\run_console.ps1 -Console <저장소>\웹_콘솔
```

## 정책 본런 결과 들여오기 (데이터를 다시 만들 때)

P010·P012·P013·거리두기·서울시 재난긴급생활비 본런은 구글 드라이브
`KW_시뮬결과_공유_20261011` 에 zip 으로 있다. 실험 정의(`web/experiments/*.json`)의
`source` 를 따라 일자 기록만 꺼내 `SIM_EXPERIMENTS_ROOT` 아래에 둔다. 다시 돌려도 안전하다.

```bash
python web/tools/import_policy_runs.py \
    --src "G:/내 드라이브/KW_시뮬결과_공유_20261011" \
    --personas "G:/내 드라이브/Kw/final_project/data/neo4j_load/agents/agents_final.json.bak_orig"
```

정책마다 개요·지도·시민 문답·보고서를 보려면 이어서 실행 폴더를 세운다(`extract_policy_runs.py` →
`build_policy_runs.py` → `scripts.sim.visualization_3d.pack` → `web/ui/tools/build_agent_console_data.py`,
각 도구 머리말에 사용법). 정책 화면·보고서는 `SIM_POLICY_DIR` 의 정책 파일을 읽는다 — 본런이 쓴 고정본
(`frozen_policy.sha256` 와 대조)을 그 폴더에 둔다. 지원금 없는 거리두기는 설명 파일 `P2020_DIST.json` 을 둔다.

실험 화면의 "정책 효과"는 같은 시민·같은 날의 정책 있음/없음 결제액 비교다.
95% 신뢰구간은 시민 단위 부트스트랩 500회(시드 20261022)로 구한다.

## 실행 파라미터 전달

콘솔은 임의 shell 명령을 받지 않는다. 운영자가 `SIM_RUN_COMMAND_JSON` 에 적은
명령만 실행하고, 화면에서 고른 값은 **환경변수로만** 자식 프로세스에 전달한다.

```
SIM_RUN_ID  SIM_POLICY_ID  SIM_START_DAY  SIM_DAYS  SIM_AGENTS
```

## API

### 정책

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/policies` | 목록 |
| `GET` | `/api/policies/next-id` | 충돌하지 않는 다음 정책 ID |
| `POST` | `/api/policies/draft/validate` | **저장하지 않고** 초안만 preflight 검증 |
| `GET` | `/api/policies/{id}` | 상세 |
| `GET`/`POST` | `/api/policies/{id}/validate` | 저장본/본문 preflight |
| `POST`/`PUT` | `/api/policies` · `/api/policies/{id}` | 저장 (preflight 통과 시에만) |
| `DELETE` | `/api/policies/{id}` | 삭제 (실행 중에는 409) |

### 실행

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/runner/lock` | lock 보유자·시작시각·프로세스 생존 여부 |
| `POST` | `/api/runner/start` | 실행 시작. `policy` 본문을 함께 보내면 **새 정책을 주입**한다 |
| `POST` | `/api/runner/stop` | 소유 프로세스에 graceful 중단 신호 |
| `POST` | `/api/runner/release` | 죽은 프로세스가 남긴 lock 해제 |

`start` 요청 본문:

```json
{
  "run_id": "BASE_0810",
  "policy_id": "P012",
  "policy": { "id": "P012", "name": "...", "decile_grants": { "1": 400000 } },
  "start_day": "2025-07-21", "days": 7, "agents": 200
}
```

`policy` 가 있으면 **먼저 저장하고**(preflight 통과 필수) 그 정책으로 실행한다.
순서를 뒤집지 않으므로 검증되지 않은 정책으로 시뮬레이션이 시작되는 경로가 없다.

### 보고서

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/reports/catalog?run_id=&policy_id=` | 엔진 가용 여부, 절 목록, 정책 결합 근거, LLM 상태 |
| `POST` | `/api/reports/jobs` | 생성 시작 (`engine`: `v2` \| `dasol`) |
| `GET` | `/api/reports/jobs` · `/api/reports/jobs/{id}` | 진행 상태·단계·로그·산출물 |
| `GET` | `/api/artifacts/{path}` | 생성된 보고서 파일 |

### LLM

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/llm/status` | 제공자·모델·설정 여부. **키 값은 절대 내보내지 않는다** |
| `POST` | `/api/llm/ping` | 실제 왕복 1회로 연결 확인 |
| `GET` | `/api/interview/status` | 인터뷰 가능 여부와 어느 백엔드로 답하는지 (`dedicated` \| `shared` \| `none`) |
| `POST` | `/api/interview` | 대상자 한 명에게 질문. 근거는 화면이 보낸 그 사람의 기록뿐 |

`status` 의 `reason` 은 **화면에 그대로 나가는 문장**이라 `.env` 같은 내부 사정을 담지
않는다. 고칠 수 있는 사람에게 필요한 안내는 `operator_hint` 로 따로 실리고, 읽기 전용
시연 모드에서는 화면에 뜨지 않는다.

### 키 한 줄로 켜지는 범위

`.env` 에 `GEMINI_API_KEY` 한 줄만 넣으면 **보고서 해설과 인터뷰가 함께** 켜진다.
인터뷰 전용 엔드포인트(`INTERVIEW_*`)가 설정돼 있으면 그쪽을 먼저 쓰고, 없을 때만
보고서 해설과 같은 클라이언트로 물러난다. 어느 쪽으로 답했는지는 응답의
`model_label` 에 남으므로 조용히 바뀌지 않는다.

## 테스트

Windows의 실행 상태 조회는 프로세스에 신호를 보내지 않고 Win32 API로 확인한다.
Windows에서는 웹 콘솔의 중단 요청을 거절하고 실행 lock을 유지한다. 운영 도구에서
정상 종료한 뒤 lock을 해제해야 한다. POSIX 환경에서는 소유 PID에 SIGINT를 보낸다.
콘솔의 lock은 콘솔이 시작한 실행에만 적용된다. 외부 CLI 실행과 함께 사용하는 경우
읽기 전용 모드(`WEB_READ_ONLY=true`)로 실행한다.

```bash
python -m pytest tests/unit/report tests/unit/test_report_api.py tests/unit/test_s2_api.py -q
cd web/ui && npx tsc --noEmit -p tsconfig.json
```
