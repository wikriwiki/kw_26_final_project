# 웹 콘솔·보고서 main 병합 검증 (2026-09-22)

기준 main: `7fc60f9f38dad80d6b5760f90b43de8ff83a92df`.
작업 브랜치: `feat/report-v2-web-console`의 `3051e95`까지 10개 커밋과
작업 폴더의 미커밋 소스·테스트·문서·배포 설정을 통합했다.

## 시뮬레이션 영향

- 기존 `scripts/sim` 파일 중 달라지는 것은 `export_visualization.py`뿐이다.
  명시적인 `--per-gu` 옵션으로 내보내기 표본 수를 바꾼다. 옵션을 생략하면
  main의 SAMPLE 표와 실행 동작이 그대로 유지된다. 데이터베이스 읽기용 내보내기다.
- 실행·계획·새벽·야간·구매·정책 지갑 코드, `scripts/policy_pipeline`,
  `scripts/persona`, `scripts/neo4j_load`, `data`, 루트 Dockerfile·운영 compose·
  requirements 및 CI workflow는 기준 main과 동일하다.
- 새 시각화 가공·검증 스크립트는 독립 실행 도구다. 기존 시뮬레이션 실행 경로에
  새 보고서나 웹 모듈 import를 추가하지 않았다.
- 웹 실행기는 운영자가 설정한 고정 명령만 실행하며, 파라미터는 자식 환경변수로
  전달한다. 보고서 해설과 인터뷰 설정은 별도 웹/보고서 모듈에 속한다.
- Windows에서 `os.kill(pid, 0)`과 `os.kill(pid, SIGINT)`가 정상 상태 조회·정상
  중단으로 동작하지 않는 문제를 수정했다. 상태 조회에는 Win32 읽기 API를 쓰고,
  Windows 웹 중단은 409로 거절하며 lock을 보존한다. POSIX 중단은 소유 PID에
  SIGINT를 보낸다. 접근 권한이 없으면 실행 중으로 간주해 lock을 보존한다.
- 콘솔 lock은 콘솔이 관리하는 실행을 대상으로 한다. 외부 CLI 실행과 병행할
  때는 `WEB_READ_ONLY=true`를 사용해야 한다. AWS compose는 읽기 전용이다.

## 검증 결과

Python 3.13 / Windows에서 영역별 별도 pytest 프로세스로 실행했다.
전체 디렉터리를 한 번에 수집하면 기존 persona/Neo4j의 `_common` 모듈 이름이
충돌하므로, 영역별 실행 결과로 기준 main과 대조했다.

| 영역 | 통과 | 실패 | 건너뜀 |
|---|---:|---:|---:|
| 시뮬레이션 `tests/unit/sim` | 470 | 3 | 0 |
| 보고서·웹·실행기 | 143 | 2 | 5 |
| 정책 파이프라인 | 63 | 0 | 0 |
| 페르소나 | 58 | 0 | 0 |
| 합계 | 734 | 5 | 5 |

보고서·웹 검사에서는 추가로 3,628개 subtest가 통과했다. 건너뛴 검사는
실제 실행 데이터가 별도 검증 작업 트리에 마운트되지 않아 실행하지 못한 검사다.
`npm ci --no-audit --no-fund`와 `npm run build`가 통과했으며,
빌드 후 SPA deep-link/API 경로 검사도 통과했다. 관련 Python 모듈 컴파일도 통과했다.

실패 5개는 변경 없는 기준 main을 별도 작업 트리에 체크아웃해 동일하게 재현했다.
기준본의 sim/report 결과는 491 passed, 5 failed였다. 새 회귀 실패는 없다.

- `test_outside_need.py`: 기존 SERVICE 후보 범위와 기대값 불일치 3개.
- `test_composition_distance.py`: bootstrap 테스트 입력 구조와 현재 구현 불일치 2개.

추가 실행기 회귀 검사 7개는 상태 조회/중단 시 Windows 테스트용 자식 프로세스가
살아 있는지, 권한 오류 시 lock 보존, 고정 명령·환경 전달, 이중 실행 차단,
시작 실패/빈 명령 처리, POSIX 소유 PID 중단, 명령 미설정 시 정책 미저장을 확인한다.

GPU·LLM·Neo4j를 연결한 새 전체 시뮬레이션 실행은 수행하지 않았다. 기존 데이터와
진행 중인 실행에 접근하거나 이를 중단하지 않았다. GitHub repository secret 목록은
조회 시 비어 있었으며, 기존 workflow의 SSH 배포는 해당 secret 없이는 건너뛴다.

## 보존 범위

원래 작업 폴더와 기존 로컬 main은 검증 중 변경하지 않았다. `.env`, 개인 설정,
임시 로그·프레젠테이션 작업 폴더·대용량 생성물은 새 커밋에 포함하지 않았다.
원래 작업 폴더의 기존 보고서 삭제도 전파하지 않아 main의 기존 산출물을 보존했다.
`.gitignore`의 양쪽 규칙과 main의 빈 테스트 패키지 파일을 보존해 병합 충돌을 해결했다.
