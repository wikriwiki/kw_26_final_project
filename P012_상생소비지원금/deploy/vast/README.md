# 금연구역 실험: Vast.ai 실행 준비

> **2026-09-23 활성 설계 변경:** 사용자가 본 실행을 노원·서초·송파 거주자 **1,154명**으로 줄였다. 새 명단·번들·오퍼 확인과 인스턴스 생성·SSH 접속은 [START_3_DISTRICTS.md](START_3_DISTRICTS.md)를 먼저 따른다. 아래의 `7500` 파일명·인원·비용 계산은 이전 설계 기록이며 현재 본 실행에 그대로 사용하지 않는다. Day 0 덤프와 308개 정책 대상 시설, 28일 일정, 모델은 유지한다. 새 3구 번들의 원격 DB·GPU 실험은 아직 실행하지 않았다.

현재 브랜치의 **기존 Neo4j 시뮬레이션 엔진**으로 정확히 **7,500명·대상 POI 308개**의 금연구역 실험을 실행하는 절차다. 사용자 요청에 따라 기간은 **2017-11-19~12-16의 28일**, 정책 시행 전 14일과 시행일 2017-12-03부터 14일로 설정한다. 명단과 POI는 준비됐다. **Day 0=2017-11-18**의 새 덤프 `baseline_28d_v1`은 정적 데이터 fingerprint 보존·재구축·내보내기·해시 확인과 실제 로컬 Windows Neo4j 5.26.0 단일 DB 재복원을 완료했고, 7,500명 본 bundle과 30명 pilot의 그래프 사전검사를 모두 통과했다. 다음 단계는 원격 Linux에서 독립 OFF/ON DB 두 개를 복원하고 SGLang/GPU health와 pilot을 확인하는 것이다. 원격 DB 쌍·GPU 실행·SSH 연결은 아직 수행하지 않았다.

2026-09-22 계정 화면에서 **Credit $400.00, Instances 0**을 확인했다. 정확한 이미지 digest·SSH·disk 100GB를 지정한 **비공개 Vast 템플릿**을 저장했으며 기록은 `local/template-exaone-sglang-v4.json`에 있다. 서버 임대와 과금은 발생시키지 않았다. 충전 잔액 $400는 전체 실행 목표 비용이 아니다. 먼저 총 $5 이내 준비·파일 전송·짧은 pilot 예산을 잡고, 실제 처리량을 확인한 다음 본 실행 시간을 정한다. 예산 검사기는 실제 결제 한도 설정 기능이 아니다.

## 모델과 GPU 선택

사용자 지시에 따라 **기존 SGLang을 유지**하고 **LG `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`**, mode `exaone_4_5`, revision `31e6a965d0661bbe4a8b895e22a77f8271772ba0`를 기본으로 사용한다. 공식 모델 설정은 `compressed-tensors`의 pack-quantized W4A16/group128이다. 이름이 AWQ여도 `--quantization awq_marlin`을 강제하지 않고 모델 metadata로 로딩한다. SGLang의 해당 compressed-tensors GPU 경로가 하드웨어에 맞는 커널을 선택하도록 한다.

GPU 한 장에서 작은 배치로 구동할 때 가중치 메모리와 대역폭을 줄일 수 있어 AWQ를 첫 선택으로 삼았다. BF16/FP8/GGUF보다 실제 시뮬 처리속도가 빠른지는 아직 측정하지 않았으며 GPU/배치/커널에 따라 달라진다. 기본은 8,192 문맥, worker 4, max sequences 4이며 **RTX 6000 Ada 48GB·호스트 RAM 64GB 이상**을 첫 pilot 후보로 권한다. 더 저렴한 A6000 48GB는 대안이며 RTX 5090 32GB는 모델 적재와 긴 요청 검증을 통과할 때 고려한다. 두 arm에는 동일 모델/revision/서빙 옵션을 적용한다. 실행 스크립트는 다른 모델 계열을 거부한다.

이전 8B 모델 실행안, v5와 근거 기록 추가 이전 v6 소스 패키지는 현재 실험에 사용하지 않는다. LG·SGLang 28일 소스 패키지명은 `source-exaone-sglang-7500-28d-reasoning-v8.tar.gz`이며, 확정한 baseline 해시와 이 문서를 포함해 생성한다. `deploy/vast/local`의 과거 오퍼와 만료된 RTX 5090 pilot 계획은 현재 임대 실행안이 아니다. `server-recommendation-20260922-2204.json`의 RTX 6000 Ada 48GB 추천도 선정 근거 기록이며 임대 전 새 견적과 비용 계획을 만들어야 한다.

**2026-09-22 22:04 KST** 조회의 RTX 6000 Ada 48GB 후보는 **$0.687407/h**(100GB 저장공간 포함), 실효 CPU 48코어·RAM 약 120GB·표시 신뢰도 99.846%였다. 48GB 메모리 여유와 호스트·네트워크 조건을 보고 선택한 후보이며 실제 처리속도가 가장 빠르다고 검증한 결과는 아니다. 4시간 + 전송 여유 $2는 약 **$4.750**로 첫 pilot 한도 $5 이내다. 당시 전송 단가 $0.01/GB에서 다운로드 50GB·회수 5GB를 가정한 순사용료 추정은 약 $3.30이다. 설치와 모델 적재도 4시간에 포함한다. 이 견적은 **22:19 KST에 만료**되었으므로 임대 전 반드시 재조회한다. 전체 7,500명·28일 실험 비용은 아래 처리량 계산으로 별도 판단한다.

30명·1일·worker 4 pilot은 초기 실행 준비와 처리 속도 확인용이다. 당구/골프 방문 사건이 드물어 정책 효과 추정 표본으로 쓰지 않는다. 초기 측정 후 더 긴 문맥·정책 시행 후 날짜·DB 증가를 포함한 대표 구간에서 처리량을 확인하고 worker 수를 조절한다. 본 실행에서도 두 arm은 동일 코호트, 동일 입력 해시, 동일 seed, 동일 모델 revision, 동일 기간을 유지한다.

### 28일 실험의 시간과 비용 계산

한 seed의 작업량은 **7,500명 × 28일 × OFF/ON 2개 = 420,000 agent-days**다. 이는 시뮬레이션 작업량이며 서버를 실제 28일 임대한다는 뜻이 아니다. 두 arm을 순차 실행하는 현재 방식에서 정상 완료 처리량이 `q agent-days/hour`라면 예상 실행 시간은 `420000 / q`시간, 사용료는 `시간 × 실제 시간당 단가 + 설치 시간 비용 + 전송 비용`이다. 여러 seed를 실행하면 작업량도 그만큼 증가한다.

아래는 **측정 결과가 아닌 가정 계산**이며 만료된 $0.687407/h를 예시 단가로 쓴다. 설치·전송·실패 재실행 비용은 별도다.

| 정상 처리량 q | 두 arm 실행 시간 | GPU·100GB 저장공간 사용료 예시 |
|---:|---:|---:|
| 500 agent-days/h | 840시간 | $577.42 |
| 1,000 agent-days/h | 420시간 | $288.71 |
| 2,000 agent-days/h | 210시간 | $144.36 |

잔액 $400로 전체 실행이 가능한지는 실측 처리량과 추가 비용을 확인해야 알 수 있다. 예를 들어 500 agent-days/h라면 사용료만으로 잔액을 넘는다. 첫 $5 pilot 후 실제 정상 agent-days/h, 오류율, DB 병목과 GPU 사용량으로 본 실행 계획을 정한다.

## 1. 과금 전 로컬 준비

저장소 루트 PowerShell에서 이미 준비된 bundle을 검사한다.

```powershell
python deploy/vast/prepare.py preflight
python scripts/experiments/no_smoking_zone.py preflight --bundle output/no_smoking_zone/full_7500_v1
python scripts/experiments/no_smoking_zone.py preflight --bundle output/no_smoking_zone/pilot_7500_v1
```

본 실험 명단은 `data/experiments/no_smoking_zone/cohort_7500_ids.json`의 **정확히 7,500명**이다. 원래 실행자의 적격한 7,494명을 유지하고 소비 기준값이 없는 6명만 적격 풀에서 SHA256 순위·seed `20171203`으로 대체했다. 흡연 여부나 결과를 대체자 선정에 사용하지 않았다. 14,549명의 적격 풀 전체를 본 실험 규모로 쓰지 않는다. POI 308개는 원본 상호·카테고리에 따른 보수적 분류이며 실제 등록업종·실내 배치·현재 영업 여부를 확인한 표본은 아니다.

새 버전의 bundle이 필요할 때만 원본 그래프와 대조한 persona·적격 명단·고정 코호트·POI를 모두 지정해 새 출력 경로로 만든다. 본 실행은 `--limit`을 넣지 않으며 pilot만 같은 명단에서 30명을 선택한다.

```powershell
python scripts/experiments/no_smoking_zone.py prepare --agents output/no_smoking_zone/graph_cohort_v2/personas.json --eligible-ids output/no_smoking_zone/graph_cohort_v2/eligible_ids.json --cohort-ids data/experiments/no_smoking_zone/cohort_7500_ids.json --pois output/no_smoking_zone/staging_audit/verified_pois.json --seed 20171203 --simulation-seed 17001 --out output/no_smoking_zone/full_7500_v2
python scripts/experiments/no_smoking_zone.py prepare --agents output/no_smoking_zone/graph_cohort_v2/personas.json --eligible-ids output/no_smoking_zone/graph_cohort_v2/eligible_ids.json --cohort-ids data/experiments/no_smoking_zone/cohort_7500_ids.json --pois output/no_smoking_zone/staging_audit/verified_pois.json --seed 20171203 --simulation-seed 17001 --limit 30 --out output/no_smoking_zone/pilot_7500_v2
```

`<...>`는 실제 경로/값으로 바꾸며 기존 출력 경로를 덮어쓰지 않는다. 사전 검사 실패를 fixture나 합성 POI로 우회하지 않는다.

DB는 기존 운영 DB를 가리키지 않는다. 덤프를 만든 **동일 Neo4j 버전**으로 전용 서버 두 개에 같은 깨끗한 Day 0 덤프를 복원한다. Community Edition이라면 독립 프로세스/독립 data 디렉터리와 포트를 쓰고, Enterprise의 독립 DB를 쓴다면 서로 다른 DB 이름을 지정한다. 덤프/POI/코호트와 엔진 schema의 일치 확인은 필수다. 초기 정책 상태·State·관계가 오염된 DB는 재사용하지 않는다.

발견한 `BASE7500H_nopolicy_7d.dump`와 중복 사본 `(1)`은 **785,727,594 bytes**, SHA256 `3417baa250f6babe4de7767fd3a2c6689450a8272f40ebef552626c4192bb263`으로 동일하며 기존 frozen 검증본과 일치한다. 이 Neo4j **5.26.0** 원본은 2025-07-14~20 실행 후 상태다. 원본을 보존하고 격리 복원본에서 정적 데이터 보존·초기 State 재구축을 완료했다.

28일 실험의 기준 덤프는 **`output/no_smoking_zone/baseline_28d_v1/neo4j.dump`**, **841,292,022 bytes**, SHA256 **`6103c534628da29c6aafdbbe808ff530af5b0e47234466eb8b90d535f979830d`**이며 Day 0은 **2017-11-18**이다. 정적 데이터 fingerprint 보존·재구축·내보내기·해시 확인 후 같은 격리 Windows 환경의 Neo4j 5.26.0 단일 DB에 실제 재복원했다. `full_7500_v1`·`pilot_7500_v1`의 `graph_preflight`는 모두 통과했으며 결과는 `output/no_smoking_zone/staging_audit/clean_reload_7500_28d_validation.json`의 `all_checks_passed=true`에 기록했다(2026-09-22 13:20:39 UTC). 기존 2017-11-25 Day 0 덤프는 28일 실험에 재사용하지 않는다. 원격 Linux의 독립 두 팔 복원은 별도 검증이 필요하다. 출처와 검사 기록은 `data/experiments/no_smoking_zone/graph_source.json`, `baseline_rebuild_validation.json`에도 남긴다. 덤프는 소스 패키지에 포함하지 않고 별도로 전송한다.

Vast의 일반 임대는 이미 Docker 컨테이너 안이다. 그 안에서 Docker Compose가 된다고 가정하지 않는다. `bootstrap_neo4j.sh`는 별도로 검증한 **깨끗한 Day 0 덤프**만 입력받아 Neo4j 5.26.0 전용 인스턴스 두 개를 복원한다. 실행 이력이 있는 원본 덤프의 정적 데이터 재구축 작업은 이 스크립트의 범위에 포함하지 않는다.

## 2. 공개 오퍼 검색과 구체적인 비용 계획

공개 검색은 로그인/CLI 설치 없이 가능하며 자원을 생성하지 않는다. 공개 접근이 바뀌면 아래 인증 CLI 경로를 사용한다.

```powershell
python deploy/vast/public_offers.py --min-gpu-gb 48 --gpu "RTX 6000Ada" --gpu "RTX A6000" --out deploy/vast/local/offers-exaone-fresh.json
```

응답의 실제 offer ID를 지정해 계획만 생성한다. 예산은 네트워크 여유분을 포함한다. `$0.75/h, 4시간, 전송 여유 $2, 총 $5`는 첫 pilot의 제안 한도다. 입력/모델 다운로드와 설치 시간도 임대 시간에 포함한다. 실제 33B 작업의 전체 소요 시간은 30명 pilot으로 측정한 뒤 정한다.

```powershell
python deploy/vast/prepare.py plan --offers deploy/vast/local/offers-exaone-fresh.json --offer-id <실제_offer_ID> --max-hourly-usd 0.75 --max-hours 4 --transfer-reserve-usd 2 --total-budget-usd 5 --out deploy/vast/local/pilot-plan-exaone.json
```

계획은 15분 넘은 견적, 시간당 한도 초과, 총예산 초과, `latest` 이미지 등을 거부한다. 외부 변경 없이 생성 명령을 보여준다. `--max-hours`는 검토용 수치이며 서버를 자동 해지하지 않는다. 모델·이미지 다운로드 크기 × `inet_down_cost`, 회수 크기 × `inet_up_cost`가 reserve를 넘으면 여유분을 늘리고 총액을 다시 계산한다. 기계별 네트워크 단가를 반드시 확인한다.

인증 CLI가 필요하면 별도 venv에 설치하고 키를 로컬 secret manager 또는 사용자 홈의 Vast 설정으로 공급한다. 키를 채팅, 명령 예시, 저장소 `.env`, GPU 서버에 복사하지 않는다.

```powershell
python -m venv .venv-vast
.venv-vast/Scripts/python -m pip install vastai
& .venv-vast/Scripts/Activate.ps1
python deploy/vast/prepare.py search
python deploy/vast/prepare.py search --execute --out deploy/vast/local/offers-cli-fresh.json
```

Vast CLI의 `VAST_API_KEY` 환경변수 또는 `~/.config/vastai/vast_api_key`를 사용한다. 준비 당시에는 CLI와 로컬 키가 없었으나 브라우저 계정 로그인과 비공개 템플릿 저장은 확인했다. 로컬 입력·덤프 준비는 완료했고, 실제 실행 전에 원격 독립 Day 0 DB 두 개와 GPU 환경을 확인해야 한다. `preflight`는 키의 존재 여부만 출력한다. 프로젝트 전용 SSH 공개키를 계정에 등록한 뒤 임대한다. 기존 개인키는 덮어쓰지 않는다.

## 3. 서버 임대와 SSH

입력/DB 준비가 끝나면 갱신된 계획의 `create_command`를 실행한다. 이 명령부터 비용이 발생한다. 공식 `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` 이미지가 존재함을 확인했고, amd64 digest `sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a`로 고정한다. bootstrap이 기존 `scripts/deploy/install_sglang_exaone45.sh`에 설치를 위임해 **LG 공식 모델 카드가 안내하는 SGLang fork commit `6757c9f904cdb8ae9028a394a2108d079b9e088c`**를 별도 venv에 설치한다. 검증된 호환 조합 `torch==2.9.1`, `transformers==5.8.0`, `kernels==0.10.0`을 유지한다. fork가 요구하는 이전 Transformers pin은 기존 설치 방식과 동일하게 `--no-deps`로 덮어쓴다.

임의의 최신 SGLang 릴리스로 바꾸지 않는다. bootstrap과 launcher는 설치된 fork commit, 위 패키지 버전, EXAONE 4.5 아키텍처 파일의 존재를 확인해 불일치 시 중단한다. 설치 자체는 최대 30분으로 제한한다. 이 조합의 근거와 이미지 실존은 확인했지만 **이번 GPU에서 실제 설치/모델 적재는 아직 실행하지 않았다**. 두 arm은 같은 이미지와 설치 pin으로 실행한다.

```text
vastai create instance OFFER_ID --image pytorch/pytorch@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a --disk 100 --ssh --direct --cancel-unavail --label No_SmokingZone_EXP --raw
vastai show instance INSTANCE_ID --raw
vastai ssh-url INSTANCE_ID
```

생성 응답의 `new_contract`가 **instance ID**다. offer ID와 구분해 로컬 계획 기록에 저장한다. `running` 상태가 되기 전에 다음 단계를 진행하지 않는다. 설치/모델 health 실패 시 재시작을 무한 반복하지 말고 로그를 회수해 인스턴스를 종료한다.

SSH 정보에서 실제 호스트와 포트를 가져와 사용한다. 첫 연결 때 호스트 키를 확인하고 이후 호스트 키 검증을 유지한다. `StrictHostKeyChecking=no`나 공개 8000 포트는 사용하지 않는다.

```powershell
ssh -i "$env:USERPROFILE/.ssh/no_smoking_vast" -p <SSH_PORT> -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 root@<SSH_HOST>
```

서버에서 실행할 시뮬레이션은 서버 내부 `127.0.0.1:8000/v1`로 추론한다. 위 터널은 로컬 점검에만 사용한다. Neo4j Bolt/HTTP 포트도 공개하지 않는다.

## 4. 현재 작업 파일과 검증한 입력 전달

새 브랜치가 아직 push되지 않아도 현재 파일 내용으로 전송할 수 있다. 패키저는 runtime 경로의 Git 추적/미추적 소스와 **`output/stats/*.json` 전체**를 포함하고 `.git`, 개인키, 실제 env, 로컬 견적, DB, 원본 첨부 문서를 제외한다. 가격·이동 기준 5종(`unit_price`, `dong_context`, `hub_catalog`, `dong_centroids`, `hub_signature`)이 없거나 통계 JSON이 손상되었거나 Git LFS 포인터이면 패키징을 중단한다. `deployment-manifest.json`에 모든 통계 파일의 SHA256와 선택 파일 `poi_menu_price.json`의 부재 여부를 기록한다. 생성된 목록/아카이브를 검토한다. 입력 bundle과 DB 덤프는 별도 전송한다. 명단이 같아도 기간 변경으로 bundle runtime 해시가 달라졌다면 `experiment-bundles-7500-v1.tar.gz`도 새 내용으로 다시 만들어 해시를 확인해야 한다.

통계 파일이 존재해도 실측 절대 업종 가격과 개별 POI 메뉴 가격은 현재 누락되어 있다. 이로 인한 합성 가격/기존 하드코딩 값의 한계는 runner의 reference 검사 결과에 남으며, 2017년 관측 가격으로 해석하지 않는다.

```powershell
# 아직 생성하지 않았을 때만 패키징한다. 기존 아카이브를 덮어쓰지 않는다.
python deploy/vast/package_source.py --out deploy/vast/local/source-exaone-sglang-7500-28d-reasoning-v8.tar.gz
tar -tzf deploy/vast/local/source-exaone-sglang-7500-28d-reasoning-v8.tar.gz
Get-FileHash deploy/vast/local/experiment-bundles-7500-v1.tar.gz -Algorithm SHA256
Get-FileHash output/no_smoking_zone/baseline_28d_v1/neo4j.dump -Algorithm SHA256
scp -i "$env:USERPROFILE/.ssh/no_smoking_vast" -P <SSH_PORT> deploy/vast/local/source-exaone-sglang-7500-28d-reasoning-v8.tar.gz deploy/vast/local/source-exaone-sglang-7500-28d-reasoning-v8.tar.gz.sha256 root@<SSH_HOST>:/workspace/
scp -i "$env:USERPROFILE/.ssh/no_smoking_vast" -P <SSH_PORT> deploy/vast/local/experiment-bundles-7500-v1.tar.gz root@<SSH_HOST>:/workspace/
scp -i "$env:USERPROFILE/.ssh/no_smoking_vast" -P <SSH_PORT> output/no_smoking_zone/baseline_28d_v1/neo4j.dump root@<SSH_HOST>:/workspace/no-smoking-baseline.dump
```

서버 SSH 세션에서 새 작업 디렉터리에 푼다.

```bash
cd /workspace
sha256sum -c source-exaone-sglang-7500-28d-reasoning-v8.tar.gz.sha256
# bundle 해시는 위 로컬 Get-FileHash 결과와 비교한 뒤에만 푼다.
sha256sum experiment-bundles-7500-v1.tar.gz
printf '%s\n' '6103c534628da29c6aafdbbe808ff530af5b0e47234466eb8b90d535f979830d  no-smoking-baseline.dump' | sha256sum -c -
mkdir no-smoking-project experiment-bundles
tar -xzf source-exaone-sglang-7500-28d-reasoning-v8.tar.gz -C no-smoking-project
tar -xzf experiment-bundles-7500-v1.tar.gz -C experiment-bundles
cd no-smoking-project
bash deploy/vast/bootstrap.sh
cp deploy/vast/.env.example deploy/vast/local.env
```

`local.env`의 두 arm DB 주소/인증, 검증한 snapshot SHA256를 채운다. 실제 비밀값을 출력하거나 셸 tracing(`set -x`)을 켜지 않는다. runner는 두 arm의 endpoint가 서로 다른지 확인하고 **현재 실행하는 arm의 DB와 `ExperimentSnapshot` 표식**을 검사한다. 반대 arm의 DB는 그 arm을 시작할 때 검사한다. 표식은 runner 문서의 계약에 맞아야 하며 확인 없이 임의 표식만 붙여 검사를 우회하지 않는다.

### 같은 Day 0에서 독립 Neo4j 두 개 복원

검증된 재구축 결과 `.dump`와 SHA256를 별도로 전송한다. 이전 7일 실행 후 원본 덤프를 여기에 사용하지 않는다. PyTorch 이미지는 Java 포함을 보장하지 않으므로 Java 17 또는 21을 확인한다. 없으면 Ubuntu 컨테이너에서 아래 패키지를 설치한다. 자동 bootstrap은 Java를 임의 설치하거나 전역 설정을 변경하지 않는다.

```bash
java -version
# Java 17/21이 없을 때만 실행한다.
apt-get update && apt-get install -y openjdk-17-jre-headless
```

`local.env`에 `CLEAN_BASELINE_DUMP=/workspace/no-smoking-baseline.dump`, `NO_SMOKING_SNAPSHOT_SHA256=6103c534628da29c6aafdbbe808ff530af5b0e47234466eb8b90d535f979830d`와 DB 암호를 채운다. 공통 `NEO4J_PASSWORD` 또는 `NO_SMOKING_OFF_NEO4J_PASSWORD`, `NO_SMOKING_ON_NEO4J_PASSWORD`를 환경변수로 전달한다. 암호를 명령행에 쓰거나 출력하지 않는다.

```bash
set -a
source deploy/vast/local.env
set +a
bash deploy/vast/bootstrap_neo4j.sh --dry-run
bash deploy/vast/bootstrap_neo4j.sh
source "$NO_SMOKING_NEO4J_ROOT/pair.env"
```

기본 설치 위치는 `/workspace/no-smoking-neo4j`이며 **기존 폴더가 있으면 빈 폴더라도 중단**한다. `/workspace` 밖이나 심볼릭 링크 경로도 거부한다. OFF/ON은 서로 다른 `off/`, `on/` home·설정·data·로그·PID 경로를 쓰며 Bolt는 각각 `127.0.0.1:17687`, `127.0.0.1:17688`에만 바인딩한다. HTTP/HTTPS는 비활성화한다. systemd나 기존 Neo4j 서비스를 변경하지 않는다.

기본 메모리는 **인스턴스당 heap 2GiB + page cache 512MiB**이며 32GiB RAM 호스트에서 SGLang과 함께 사용하는 시작값이다. `NO_SMOKING_NEO4J_HEAP_MB`, `NO_SMOKING_NEO4J_PAGECACHE_MB`로 조절한다. 복원 전 최소 여유 디스크 20GiB, 포트 충돌, 원본 해시를 검사한다. dry-run은 이 검사를 수행하며 다운로드·복원·파일 생성은 하지 않는다. 실제 실행은 Python Neo4j 드라이버와 Java 버전을 추가 확인한다.

공식 Linux 배포물을 SHA256 `ad8ac3398606145502b8f489530bbd39333707ae4668156af16b6086b5b037d7`로 고정한다. 원본 덤프는 읽기 전용 스트림으로 두 번 복원하고 덮어쓰기를 금지한다. 인증을 켠 새 DB의 초기 암호를 localhost 연결 직후 드라이버로 바꾸므로 비밀을 프로세스 인자에 넣지 않는다. 두 DB의 `ExperimentSnapshot.id`와 `sha256`를 최종 기준 덤프 해시로 기록하고 확인한 뒤에만 `pair-manifest.json`과 비밀 없는 `pair.env`를 생성한다. 실패하면 이 스크립트가 시작한 프로세스를 중지하고 진단 파일을 보존한다. 실패 폴더를 재사용하지 않는다.

pilot 후 새 쌍을 준비할 때는 기존 두 프로세스를 각 home의 `bin/neo4j stop`으로 중지하고 **새 root 경로**를 지정한다. 동일 포트를 다시 사용할 수 있는지 dry-run으로 확인한다. 삭제나 기존 DB 덮어쓰기는 이 도구에서 제공하지 않는다.

근거: [공식 Linux 배포물](https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz), [공식 체크섬](https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz.sha256), [Neo4j 5.26 Java 17/21 지원](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/breaking-changes/), [dump load와 기본 덮어쓰기 금지](https://neo4j.com/docs/operations-manual/current/backup-restore/restore-dump/).

## 5. SGLang 시작, pilot, 두 arm 실행

서버에서 환경을 로드하고 명령을 먼저 확인한다. SGLang 프로세스는 포그라운드에서 시작하고 다른 SSH/tmux 창으로 다음 작업을 진행하면 종료·오류를 확인하기 쉽다.

```bash
set -a
source deploy/vast/local.env
set +a
source "$NO_SMOKING_NEO4J_ROOT/pair.env"
bash deploy/vast/serve_sglang.sh --dry-run
bash deploy/vast/serve_sglang.sh 2>&1 | tee /workspace/sglang-no-smoking.log
```

두 번째 서버 터미널에서 동일한 환경을 읽는다. 기본 1시간 timeout은 **시뮬 프로세스만** 중단시킨다. 모델 프로세스와 Vast 과금은 계속되므로 인스턴스 관리 시간을 따로 확인한다.

```bash
cd /workspace/no-smoking-project
set -a
source deploy/vast/local.env
set +a
source "$NO_SMOKING_NEO4J_ROOT/pair.env"
export BUNDLE_DIR=/workspace/experiment-bundles/pilot_7500_v1
export ARM=off START_DATE=2017-11-19 DAYS=1 WORKERS=4
export OUTPUT_DIR=/workspace/no-smoking-results/pilot-off
bash deploy/vast/run_experiment.sh --dry-run
bash deploy/vast/run_experiment.sh
```

runner preflight → `/health` → 모델 일치 검사 → 실제 짧은 추론을 통과해야 시작한다. 서빙 스크립트는 SGLang·Transformers 버전과 model revision 및 실행 인자를 `runtime/server-config.json`에 기록하고 실행 wrapper는 그 경로를 `NO_SMOKING_SERVER_CONFIG`로 전달한다. runner는 이 서버 설정과 통계 입력 해시를 두 arm 사이에 비교한다. pilot의 정상 agent-days/hour, 실제 토큰/초, 실패·retry·fallback 비율, DB 응답 시간을 기록한다. `hourly rate / 정상 agent-days/hour`로 비교하고, GPU idle이 크면 더 비싼 GPU보다 DB/I/O 병목을 먼저 해결한다. 근거 데이터로 기대 정책 효과에 맞도록 모델을 선택하지 않는다.

pilot DB는 실행 중 상태가 바뀌므로 **본 실행 전 두 arm 모두 동일 Day 0=2017-11-18에서 새로 복원**한다. 본 실행은 본 실험용 bundle로 설정한다. 기간은 **2017-11-19~12-16**이며 시행 전 **11-19~12-02의 14일**, 시행일부터 **12-03~12-16의 14일**이다. ON 정책 활성일은 2017-12-03으로 유지하고 OFF는 전체 기간 정책을 비활성화한다.

```bash
export BUNDLE_DIR=/workspace/experiment-bundles/full_7500_v1
export DAYS=28 START_DATE=2017-11-19 WORKERS=4
# 처리량/비용 측정 후 전체 실행 예상 시간에 맞춰 설정한다.
export RUN_TIMEOUT_SECONDS=<확정한_초_단위_실행_한도>
export ARM=off OUTPUT_DIR=/workspace/no-smoking-results/off
bash deploy/vast/run_experiment.sh
export ARM=on OUTPUT_DIR=/workspace/no-smoking-results/on
bash deploy/vast/run_experiment.sh
.venv-no-smoking/bin/python scripts/experiments/no_smoking_zone.py score --off /workspace/no-smoking-results/off --on /workspace/no-smoking-results/on --out /workspace/no-smoking-results/report.json
```

오류가 났거나 timeout 코드 124로 끝나면 결과를 완료로 표시하지 않는다. 실패 출력을 보존하고 DB 상태를 확인한다. 기존 arm의 출력 폴더에 새 실행을 덮어쓰지 않는다.

## 6. 결과 회수와 과금 종료

서버에서 결과/로그/환경 버전만 아카이브한다. `local.env`, API 키, DB 암호는 포함하지 않는다. 모델 다운로드 cache도 회수하지 않는다.

```bash
cd /workspace
tar -czf no-smoking-results.tar.gz no-smoking-results no-smoking-project/output/experiments/no_smoking_zone/runtime sglang-no-smoking.log
sha256sum no-smoking-results.tar.gz > no-smoking-results.tar.gz.sha256
```

로컬 PowerShell에서 회수하고 해시와 내용 열람까지 확인한다.

```powershell
scp -i "$env:USERPROFILE/.ssh/no_smoking_vast" -P <SSH_PORT> root@<SSH_HOST>:/workspace/no-smoking-results.tar.gz root@<SSH_HOST>:/workspace/no-smoking-results.tar.gz.sha256 deploy/vast/local/
Get-FileHash deploy/vast/local/no-smoking-results.tar.gz -Algorithm SHA256
Get-Content deploy/vast/local/no-smoking-results.tar.gz.sha256
tar -tzf deploy/vast/local/no-smoking-results.tar.gz
vastai destroy instance <INSTANCE_ID>
vastai show instances --raw
```

회수 검증 후 **기록해 둔 정확한 instance ID 하나만** destroy한다. stop은 저장공간 비용을 끝내지 않는다. destroy는 컨테이너 데이터를 지우므로 회수 성공 전에 실행하지 않는다. 프로세스가 끝났거나 SSH가 끊긴 것은 임대 종료가 아니다. CLI/API 응답이 애매하면 새로 만들거나 중복 destroy하기 전에 해당 instance 상태를 확인한다.

## 공식 근거와 검증 범위

- [Vast 오퍼 검색과 storage 가격 옵션](https://docs.vast.ai/cli/reference/search-offers)
- [공식 CLI 공개 검색 구현](https://github.com/vast-ai/vast-cli/blob/master/vast.py)
- [Vast create instance: SSH/direct/cancel-unavail](https://docs.vast.ai/cli/reference/create-instance)
- [SSH 연결·포트 터널·SCP](https://docs.vast.ai/guides/instances/connect/ssh)
- [CLI 인증과 키 저장 위치](https://docs.vast.ai/cli/authentication)
- [컨테이너 저장공간과 종료 후 과금](https://docs.vast.ai/guides/instances/storage/types)
- [LG EXAONE 4.5 33B AWQ 공식 설정](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-AWQ/blob/31e6a965d0661bbe4a8b895e22a77f8271772ba0/config.json)
- [LG 공식 모델 카드의 SGLang 안내](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-AWQ/blob/31e6a965d0661bbe4a8b895e22a77f8271772ba0/README.md), [고정한 EXAONE 지원 fork](https://github.com/lkm2835/sglang/tree/6757c9f904cdb8ae9028a394a2108d079b9e088c)
- [PyTorch 공식 컨테이너 태그](https://hub.docker.com/layers/pytorch/pytorch/2.9.1-cuda12.8-cudnn9-devel/images/sha256-39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a)

이전 v7 소스의 시뮬레이션 단위 테스트는 **862개 통과·7개 건너뜀·기존 실패 3개**, persona/배포 검사는 **36개와 하위 검사 7개 통과**다. 실패 3개는 메인 커밋을 격리해도 재현되는 외출 필요도 테스트이며, 변경 전후 증거를 `validation_prompt_evidence_stance.json`에 기록했다. 배포 검사는 28일 기본값과 1일 pilot override를 포함한다. **2017-11-18 기준 덤프의 정적 데이터 보존·재구축·내보내기·해시 확인, 실제 로컬 Windows 단일 DB 재복원과 7,500명/30명 그래프 사전검사는 앞서 통과**했다. 원격 Linux OFF/ON 쌍 복원·GPU 모델 적재·SSH 연결·전체 정책 시뮬레이션은 아직 수행하지 않았다. 런타임 의존성은 SGLang 서버 venv를 바꾸지 않는 별도 client venv에 설치하며, bootstrap이 실측 패키지 목록을 결과에 남긴다.

## 프롬프트·근거 기록·후속 인터뷰

현재 실행은 no_smoking_v1을 고정하고 LG tokenizer 해시와 실제 호출 토큰 예산을 확인한다. bootstrap은 client venv에 고정 Transformers와 tokenizer 3개 파일만 추가한다. 모델 입출력·근거 로그는 기본 필수이며, 인터뷰·찬반 분석 절차는 data/experiments/no_smoking_zone/interview_and_stance.md를 따른다. 전체 원문 로그를 수집하므로 첫 pilot에서 일자당 증거 파일 용량을 측정해 본 실행 디스크를 정한다. 100GB의 충분성을 가정하지 않는다. 이전 214개 테스트·단일 DB 복원 검증은 당시 소스 기준 기록이며, 현재 추가 기능은 별도 검증 보고서로 구분한다.


## v8 개인 상황과 공개 입장 품질

개인 상황을 근거로 정책 입장을 설명하는 v2 응답 계약과 기간·경험 종류를 나눈 입력 선택을 적용했다. 현재 검증 결과는 `data/experiments/no_smoking_zone/validation_reasoning_quality_v2.json`, 실행/검토 기준은 `data/experiments/no_smoking_zone/reasoning_quality_v2.md`를 따른다. v7 아카이브는 보존하되 신규 실행에 사용하지 않는다.

LG 서버 준비 후 `probe_stance_reasoning.py --execute`로 합성 사례 네 개의 실제 응답을 확인할 수 있다. 특정 찬반을 정답으로 두지 않고 자신의 상황·인용 근거·중요도·조건·불확실성의 연결을 검토한다. 구조 검사만 통과했다고 논리 품질을 통과 처리하지 않는다. 이 진단은 본 7,500명 결과에 섞지 않는다. 실제 인터뷰는 출력 예약이 1,600토큰이므로 변경된 계약으로 처리량과 비용을 다시 측정한다.
