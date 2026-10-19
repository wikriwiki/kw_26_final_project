# Vast 시작 안내: 노원·서초·송파 1,154명

이 문서는 **새 인스턴스를 임대하기 전부터 SSH 연결 확인까지**의 짧은 순서다. 뒤의 서버 설치·실험 실행 명령은 [README.md](README.md)에 있다. 기존 7,500명 아카이브 대신 `3gu-1154` 아카이브를 쓴다.

## 현재 준비된 입력

- 기존 7,500명 고정 명단에서 **노원 387명·서초 322명·송파 445명**, 합계 1,154명을 거주지로 선택했다. 기존 흡연 라벨을 유지한다. 30명 pilot은 이 명단의 부분집합이다.
- 정책 대상 POI는 기존 3개 구의 308곳이다. Day 0은 2017-11-18, 실험은 2017-11-19~12-16의 28일이며 OFF/ON은 서로 다른 Neo4j 복원본을 사용한다.
- 기존 7,500명 실험은 보존된 이전 설계다. 이 표본은 현대 합성 페르소나의 부분집합이므로 2017년 3개 구 주민 전체를 대표한다고 주장하지 않는다.
- 현재 선택 범위는 **3개 구 거주 참여자와 정책 대상 시설**이다. 기존 통근·일상 이동을 유지하므로 다른 구의 직장·일상 방문은 가능하다. 모든 이동을 3개 구로 제한하는 별도 설계는 이 입력에 적용되지 않았다.

## 1. 계정에 SSH 공개키 등록

현재 Windows 네트워크에서는 외부 22번 SSH 연결이 실패했다. Vast SSH는 인스턴스에 표시된 별도 호스트·포트를 사용하므로 그 포트를 직접 확인한다. 먼저 PowerShell에서 프로젝트 전용 키를 만든다. 같은 경로에 키가 있으면 **덮어쓰지 않는다**.

```powershell
$key = "$env:USERPROFILE/.ssh/no_smoking_vast"
if (-not (Test-Path $key)) { ssh-keygen -t ed25519 -f $key }
Get-Content "$key.pub"
```

출력된 **공개키만** Vast 계정의 SSH Keys에 등록한다. 개인키는 전송하지 않는다. Vast는 계정에 새로 등록한 키를 이후 생성하는 인스턴스에 적용한다.

## 2. 새 오퍼 확인과 첫 pilot 비용 계산

CLI는 이 작업 폴더의 `.venv-vast`에 설치돼 있다. Vast API 키는 사용자 홈의 `~/.config/vastai/vast_api_key` 또는 `VAST_API_KEY`로 공급한다. 채팅·저장소·원격 서버에 키 값을 쓰지 않는다.

```powershell
& .venv-vast/Scripts/Activate.ps1
python deploy/vast/prepare.py preflight
python deploy/vast/public_offers.py --min-gpu-gb 48 --gpu "RTX A6000" --gpu "RTX 6000Ada" --out deploy/vast/local/offers-3gu-fresh.json
python deploy/vast/prepare.py plan --offers deploy/vast/local/offers-3gu-fresh.json --offer-id OFFER_ID --max-hourly-usd 0.50 --max-hours 6 --transfer-reserve-usd 2 --total-budget-usd 5 --out deploy/vast/local/pilot-plan-3gu-fresh.json
```

`OFFER_ID`는 방금 조회한 A6000 48GB 오퍼의 숫자 ID로 바꾼다. 파일명이 이미 있으면 새 이름을 쓴다. 계획기는 **15분 지난 견적을 거부**한다. 2026-09-23 10:47 UTC 조회에서는 RTX A6000 48GB, RAM 약 64GB, 실효 CPU 8코어, 100GB 저장공간 포함 약 **$0.428/시간**이 첫 후보였다. 이 가격과 오퍼 ID는 현재 구매 보증이 아니다. 첫 6시간 계산은 약 $2.57 사용료 + $2 전송 여유 = **$4.57**이며 자동 결제 한도가 아니다. 모델 적재·실험 처리량은 아직 실측하지 않았다.

## 3. 인스턴스 생성

계획 파일의 `create_command`를 읽고 **당시 오퍼·가격을 다시 확인한 뒤** PowerShell에 복사해 실행한다. 이 시점부터 비용이 발생한다.

```powershell
(Get-Content deploy/vast/local/pilot-plan-3gu-fresh.json | ConvertFrom-Json).create_command
# 위에 출력된 명령을 확인한 다음 직접 실행
vastai show instances --raw
```

`create instance` 응답의 `new_contract`가 **instance ID**다. 오퍼 ID와 혼동하지 않는다. `running` 상태를 확인하고 Vast 화면의 SSH 아이콘 또는 `vastai ssh-url INSTANCE_ID`에서 실제 호스트와 포트를 받는다.

## 4. 이 네트워크에서 실제 Vast SSH 포트 확인

Vast의 실제 SSH 호스트·포트로 아래를 실행한다. 기본값 22를 추측해 넣지 않는다.

```powershell
$sshHost = "VAST_SSH_HOST"
$sshPort = 12345  # Vast 화면의 실제 숫자로 교체
Test-NetConnection -ComputerName $sshHost -Port $sshPort
ssh -i "$env:USERPROFILE/.ssh/no_smoking_vast" -p $sshPort -o ConnectTimeout=10 root@$sshHost
```

`TcpTestSucceeded=False`면 그 포트로는 파일 전송도 안 된다. Vast 화면의 **proxy SSH** 호스트·포트를 따로 시험한다. 둘 다 실패하면 브라우저 Jupyter Terminal 또는 다른 네트워크에서 접속을 확인한다. SSH 호스트 키는 Vast 화면의 값과 확인한다. 인증 오류라면 등록 공개키와 `-i` 개인키가 짝인지 확인한다.

## 5. 연결 후 입력 전송과 첫 실행

저장소 루트의 아래 파일 3개를 `scp -P $sshPort`로 `/workspace/`에 전송한다. 소스·번들 해시는 각각 `.sha256`과 manifest를 확인하고, Day 0 덤프는 아래 고정 해시와 비교한다.

1. `deploy/vast/local/source-exaone-sglang-3gu-1154-v9.tar.gz` 및 `.sha256`
2. `deploy/vast/local/experiment-bundles-3gu-1154-v1.tar.gz`
3. `output/no_smoking_zone/baseline_28d_v1/neo4j.dump` → 서버 `/workspace/no-smoking-baseline.dump`, SHA256 `6103c534628da29c6aafdbbe808ff530af5b0e47234466eb8b90d535f979830d`

서버에서 소스와 번들을 **서로 다른 디렉터리**에 풀고 `bash deploy/vast/bootstrap.sh`, `bash deploy/vast/bootstrap_neo4j.sh --dry-run`, 실제 복원, `bash deploy/vast/serve_sglang.sh` 순서로 진행한다. 상세 명령은 [README.md](README.md) 4~6절의 기존 `7500` 파일명을 위 `3gu-1154` 파일명으로 바꿔 사용한다. 파일럿은 `BUNDLE_DIR=/workspace/experiment-bundles/pilot_3gu_1154_v1`, `DAYS=1`, `ARM=off`로 시작한다. 파일럿 후 두 DB를 다시 깨끗한 Day 0에서 복원해야 본 실행을 시작할 수 있다.

서버 파일을 로컬로 회수하고 SHA256과 내용을 확인한 뒤 **정확한 instance ID만** `vastai destroy instance INSTANCE_ID`로 종료한다. `stop`은 데이터를 보존하며 저장공간 비용이 남는다.
