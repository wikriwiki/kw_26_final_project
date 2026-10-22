# 웹 콘솔 인수인계 — 실험 레지스트리·실제 데이터·인터뷰·3D 지도 최적화

> 마지막 갱신: 2026-10-06 KST · 브랜치 `claude/web-experiment-console` (기준 `origin/main@e3ee8802`)
> 커밋: `5a181888`(3D 지도 최적화) → `2a4b8f6f`(실험 API) → `3786b567`(실험 화면) → 이 문서. **원격에 푸시하지 않았다.**

## 1. 사용자 요청

- "이젠 우리 시뮬레이션 웹 더 제대로 구현해."
- 범위 답변: "앞으로 실험할 실험 모두 한 콘솔에서 관리" / "시뮬레이션 속 에이전트에게 실시간 인터뷰 구현 및 집단 에이전트에게 쿼리 기능 구현, 실제 데이터 연결, 결과 분석·시각화"
- 중간 우선순위: "특히 에이전트의 생활 경로를 시각화하는 페이지를 최적화해줘. 지금은 너무 로딩이 길고 버벅거려."
- 진행 중인 금연 실험(`No_SmokingZone_EXP`, Vast)은 건드리지 않는다는 기존 지시가 그대로 유효하다 — 이번 작업은 서버에서 **읽기만** 했다.

## 2. 3D 생활 경로 지도 최적화 (구현·오프라인 검사·실측 완료)

원인(실측): 데이터를 JS 리터럴로 박은 133.6MB 단일 HTML(파싱 8초, 부팅 힙 356MB) + 매 애니메이션 프레임마다
대상자 객체를 새로 만들고 모든 레이어 data 를 갈아끼움. deck.gl 9.3.3 HeatmapLayer 는 data 가 바뀌면 가중치 변환을
새로 만들고 2048² 텍스처·최대값 패스(4M 점)를 다시 그린다 — 이것이 매 프레임 일어났다.

바꾼 것
- `scripts/sim/visualization_3d/pack.py` — 데이터 팩: `manifest.json` + `agents.json.gz` + 하루치 `timeline/NNN.bin.gz`
  (열: x,y float32 / 사전 코드 / sat float32 / spent int32) + 대상자 묶음별 `detail/NNN.json.gz`(기억·일정, 고를 때만)
  + `extra.json.gz`(런타임 미사용 파생 레이어). 결정적 gzip(mtime 0), 파일별 크기·sha256, 다 쓴 뒤 교체.
  `memories.visited` 는 `memories` 의 type=visited 복사본이라(1,825/1,825 일치) 묶음에서 뺐다. 합계는 manifest 에.
- 런타임(`static/*.js`) — typed array 미리 할당, 시(hour) 바뀔 때만 색·반지름·히트맵·궤적 계산, 매 프레임은 위치 보간만.
  위치는 Float64 binary 속성(Float32 를 넘기면 deck 9.3.3 이 low 파트를 다음 정점에서 읽는다 — 확인함).
  카메라 이동 중 레이어 재구성 제거(zoomend/moveend 에서만), 히트맵 텍스처 1024, 스타일만 오면 바로 그림.
  새로 보이는 것: 고른 사람의 하루 경로, 점 툴팁, 궤적 끝이 움직이는 점을 따라감(투명도 0.6→0.45 로 밝기 맞춤).
- 단일 HTML 빌더: 팩을 `<script type="application/octet-stream">` base64 로 싣는다(실행 안 됨). `--no-detail`, `--pack-dir`.
- 서버 `web/api/viz.py`: `/viz/shell.html`(?pack= ?theme=console), `/viz/static/*`, `/viz/packs/<팩>/*`(해시 주소 immutable),
  `/api/viz/packs`. 콘솔 지도 화면은 팩을 고르고, 같은 실행의 팩이 없으면 표본임을 밝힌다. vite 프록시 추가.

실측(같은 PC, Intel Iris Xe, headless Chromium + D3D11, 시연 표본 1,825명×120프레임)

| | 예전 133.6MB | 새 단일 16.7MB | 새 껍데기+팩 |
|---|---|---|---|
| 재생(거리) | 6.1fps | 44.9fps | 48.8~49.4fps |
| 재생(도시) | 8.0fps | 42.5fps | 46.4~47.8fps |
| 정지+매 프레임 줌 | 7.2fps | 31.6fps | 45~47.4fps |
| 부팅 직후 힙 | 356MB | 62MB | 42~74MB(GC 후) |
| 부팅→조작 가능 | 지도 load 포함 7초+ | 0.9초 | 0.5~1.2초 |

화면 대조: 도시·거리·선택 세 장면을 예전과 같은 프레임·카메라로 찍어 점 위치·색·히트맵 일치 확인.
산출물(이 PC, gitignore): `web/viz_store/demo/sim_standalone.html`(16.7MB), `web/viz_store/packs/SEOUL7500/`.

## 3. 실험 레지스트리 + 실제 데이터 (구현·실데이터 확인 완료)

- 정의: `web/experiments/no_smoking_3gu_v22.json` — 팔 `shared-pre`(공통 사전 14일) → `post-off`/`post-on`(시행 후 14일),
  분석 규칙(시설 별칭 screen_golf→indoor_golf, 구 이름, 해석 한계). 접속 정보는 정의에 없다.
- 이 PC 설정(`web/.env`, gitignore): `SIM_REMOTE_NO_SMOKING_VAST="ssh -i <키 경로> -p 45025 root@<서버>"`. 키 값·암호 없음.
- 미러: 기본 `~/sim_data/experiments`(= `C:\Users\srdyh\sim_data\experiments`, OneDrive 밖). `SIM_EXPERIMENTS_ROOT` 로 바꾼다.
- 동기화 `python -m web.api.experiment_sync --experiment no_smoking_3gu_v22` 또는 콘솔 "지금 받기".
  서버 명령은 `find`·`cat`·`tail -c`·`head -c|sha256sum`·`tar -c` 뿐. 커지는 날짜 파일은 앞부분 sha256 대조 후 뒤만 받는다.
  첫 동기화 2026-10-06 03:16~03:27 UTC, 322.5MB(압축 해제 기준), 10분 18초. 인터뷰 근거 저널은 그 사람을 물을 때만(한 사람 26개 4.4MB).
- 실데이터 확인(2026-10-06 03:27 UTC 미러): shared-pre 14/14일 완료·skipped 409, post-off 11일 완료 + 12-14 진행(374명)·skipped 49,
  post-on 아직 없음. **26일치 metrics 봉인 검증 실패 0**. Day2(11-20) skipped 124 는 실험 인수인계 기록과 일치. 명단 1,154명(번들).
- 비교: `no_smoking_zone.py score` 와 같은 묶음·결측 규칙의 **미리보기**(공식 점수 아님). post-on 이 생기면 자동으로 채워진다.

## 4. 근거 인터뷰·집단 질의 (구현·가짜 모델로만 확인 — 실제 모델 미실행)

- `experiment_interview.py`: 고른 날까지의 봉인 metrics(영수증·하루 상태·정책 대상 시설) + 받아 둔 저널(실행 이벤트·공개 이유)을
  근거로 [E#] 인용. 보여 주지 않은 번호·근거에 없는 숫자는 경고. 모델 진술은 "확인되지 않은 진술"로 분리.
  시뮬레이션 자체 프롬프트의 근거 표기(`[E0033]`)는 `(입력 사실 33)` 으로 바꿔 콘솔 인용과 섞이지 않게 했다(실데이터에서 발견해 수정).
- 콘솔 대화는 `<미러>/<실험>/_console/interviews.jsonl` 에만 남는다. 실험의 사후 인터뷰(봉인 기록)와 무관하고 되먹이지 않는다.
- `experiment_query.py`: 조건별 기록 통계(LLM 없음) + 표본 집단 인터뷰(최대 30명, 병렬 3, 끝나면 미검증 요약).
- **대화 모델이 이 PC 에 연결돼 있지 않다**(`/api/interview/status` ready=false). 확인은 OpenAI 호환 가짜 서버로만 했고,
  그 테스트 기록은 미러에서 지웠다. 연결 방법(사용자 결정 필요):
  1) `GEMINI_API_KEY` (보고서 해설과 공용) 2) `INTERVIEW_BASE_URL/_MODEL/_API_KEY` 전용 엔드포인트
  3) 실험과 같은 EXAONE 을 SSH 터널로(`INTERVIEW_BASE_URL=http://127.0.0.1:<로컬포트>/v1`, 키 불필요) — **진행 중인 실험과 GPU 를 나눠 쓰므로 사용자 승인 전 쓰지 않는다.**
- 수정: `.env` 의 `INTERVIEW_TIMEOUT/MAX_TOKENS` 가 무시되던 문제(임포트 시점 읽기) 고침. `interview.chat()` 공용화.

## 5. 콘솔 화면 (구현·headless 실데이터 확인)

홈 "실험" 목록 → `/experiments/:id` 개요(팔 상태·동기화) · 결과(일별 추이 6종, OFF/ON 비교) · 대상자(필터 명단·하루 기록·인터뷰) · 집단 질의.
headless 로 실데이터 미러 + 가짜 모델에 대해 홈→개요→결과→대상자 인터뷰→집단 통계·인터뷰 흐름 확인, 콘솔 오류 0, `tsc` 통과.

## 6. 실제로 실행한 검사

- `tests/unit/sim/test_viz_pack.py`, `test_build_standalone_html_3d.py`, `tests/unit/test_viz_api.py`, `tests/unit/test_experiments_api.py`(17),
  `test_s2_api.py`, `test_report_api.py`: 함께 돌려 통과(마지막 묶음 59 passed / 3 skipped — skipped 는 UI 빌드가 없을 때의 SPA 테스트).
- `tests/unit/report` 의 **원래 있던 실패 2건**(`test_composition_distance.py` — `scripts/report/pool_indicators.py:85` AttributeError)은
  이번 변경과 무관(해당 파일 미수정). 고치지 않았다.
- 실행하지 않은 것: 실제 LLM 인터뷰, post-on 팔이 있는 비교, 실험 팔의 3D 지도(아래 7), 원격 푸시·PR.

## 7. 남은 일 / 다음 순서

1. 대화 모델 연결 방식 결정(4절) → 실제 모델로 1대1·집단 인터뷰 품질 확인(근거 번호 준수율, 숫자 경고율).
2. post-on 팔이 시작되면 "지금 받기" → 결과 탭 비교가 중간 집계로 채워지는지 확인. 두 팔이 끝나면 공식 점수는 실험 코드 `score` 로.
3. 실험 팔의 생활 경로 3D 지도: 영수증의 POI 에 좌표가 필요하다. 번들에는 정책 대상 308곳뿐이라 일반 POI 좌표 출처(그래프 덤프 등)를 정해야 한다.
4. 기존 run 화면(SEOUL7500 픽스처)을 API 기반으로 옮기기, 탐색 보고서의 버그 목록(1대1 인터뷰 히스토리 순서 등) 정리.
5. 읽기 전용 배포에서 POST 가 막히므로 인터뷰·동기화도 막힌다(현재 정책 유지). 공개 시연 정책을 정해야 한다.
