# GPU LIVE 서버 세팅 가이드 — 우리 프로젝트 배포

> 작성일: 2026-07-15 · 대상: 루키트랙 GPU LIVE 콘솔에서 이 시뮬을 구동하려는 팀원
> 전제: GPU LIVE는 "GPU 워크로드 하나 = SSH/Jupyter로 붙는 리눅스 컨테이너" 모델 (docker-compose 아님)

---

## 0. 먼저 이해할 3가지 (안 그러면 헤맴)

1. **워크로드 = GPU 컨테이너 1개.** 콘솔에서 GPU를 골라 워크로드를 만들면, SSH·Jupyter로 붙을 수 있는
   리눅스 환경이 뜬다. 우리가 로컬에서 쓰던 `docker-compose`(여러 컨테이너)를 그대로 못 쓴다 →
   **필요한 것(LLM·Neo4j·시뮬)을 그 한 컨테이너 안에서 프로세스로 띄운다.**
2. **워크로드 볼륨은 종료 시 사라진다.** 반드시 **스토리지(NAS)를 마운트**하고, Neo4j 데이터·
   시뮬 산출물·모델 캐시를 거기 둔다. 스토리지에 없는 건 워크로드 삭제 시 복구 불가.
3. **우리 프로젝트에서 GPU가 필요한 건 LLM 서버뿐.** 시뮬 코드와 Neo4j는 CPU만 쓴다.
   즉 GPU 워크로드 = "Qwen을 vLLM으로 서빙하는 자리" + 거기에 Neo4j·시뮬을 얹는다.

### 우리 프로젝트 3요소 ↔ GPU LIVE 매핑
| 요소 | 자원 | GPU LIVE에서 |
|---|---|---|
| **LLM 서버** (vLLM, Qwen3) | **GPU** | 워크로드에서 `scripts/serve/*.sh` 실행 → localhost:30000(또는 8000) |
| **Neo4j** (그래프 DB) | CPU·디스크 | 같은 워크로드에 바이너리로 기동 → localhost:7687, data는 스토리지 |
| **시뮬** (run_simulation.py) | CPU | 같은 워크로드에서 실행, 위 둘에 localhost로 연결 |

---

## 1. 배포 전략 — 두 가지, A부터 권장

- **방법 A (인터랙티브, 처음 권장)**: 기본 이미지로 워크로드를 띄우고 SSH로 붙어 그 안에서 설치·실행.
  유연하고 빠르게 시작. 아래 §2가 이것.
- **방법 B (사용자 이미지, 반복 운영 시)**: GPU LIVE 규약을 지킨 커스텀 이미지를 빌드·push해 두면
  워크로드 생성 시 바로 실행환경으로 선택. 매번 설치 안 해도 됨. §4 참조.

---

## 2. 방법 A — 인터랙티브로 시작 (단계별)

### Step 1. 워크로드 생성
콘솔 → 프로젝트 영역 → **워크로드 → [워크로드 생성]**
- **01 GPU & 리소스**
  - GPU 유형: **Qwen3-32B-AWQ면 A100 80GB급 1장**(또는 프로젝트 쿼터 최대). 8B면 24GB급도 가능
  - 워크로드 유형: **단일 노드 / 인터랙티브** (SSH·Jupyter로 붙어 작업)
  - 공유 메모리: 넉넉히(vLLM은 shm 사용) — 가능하면 16GB↑
  - 실행 환경: **기본 이미지 중 NVIDIA PyTorch 계열** (CUDA 포함) 선택
- **02 추가 설정**
  - **스토리지 마운트**: 미리 만든 스토리지를 `/home/ubuntu/workspace` 또는 `/data`에 마운트
    (Step 0에서 [스토리지 생성]으로 NAS 하나 만들어 두기 — 상태가 **BOUND**여야 마운트 가능)
  - 환경변수: 지금은 비워도 됨 (아래에서 export)
- **03 검토** → 생성. 상태가 **실행 중**이 되면 접속.

### Step 2. 접속
워크로드 → 인스턴스 클릭 → 인스턴스 상세에서:
- **SSH**: [PEM 키 다운로드] 후, 표시된 명령 그대로 —
  `ssh -i {키.pem} ubuntu@proxy-app.{도메인} -p {포트}`
- **Jupyter**: [Jupyter Lab 열기] → 최초 토큰 `ubuntu` (또는 상세에 표시된 토큰)
- 파일 업로드: **SFTP**(FileZilla) 또는 스토리지 상세의 **Web SFTP**

### Step 3. 코드·데이터 올리기 (스토리지 안에서)
스토리지 마운트 경로(예: `/data`)로 이동해 거기서 작업 — 워크로드 삭제돼도 보존됨.
```bash
cd /data                       # 스토리지 마운트 경로
git clone https://github.com/wikriwiki/kw_26_final_project.git
cd kw_26_final_project
git checkout main              # 최신 (리뷰계측+가격/쿠폰/봉투 병합본, e0d4814)
```
Neo4j 덤프(`neo4j_*.dump`)와 대용량 입력은 git에 없으므로 **SFTP/Web SFTP로 `/data`에 업로드**.

### Step 4. LLM 서버 기동 (GPU) — background
```bash
# 파이썬 가상환경 권장
python3 -m venv /data/venv && source /data/venv/bin/activate
pip install -r requirements.txt          # 시뮬 런타임 의존성

# 모델 캐시를 스토리지에 (재다운로드 방지 — 워크로드 재생성해도 유지)
export HF_HOME=/data/hf_cache

# LLM 서버 실행 (기본 qwen8b, 32B는 serve_qwen32b.sh). background + 로그
bash scripts/serve/run_vllm_qwen3_8b_awq.sh > /data/llm.log 2>&1 &
# 준비 확인 (포트는 스크립트 기준 — SGLang 30000 / vLLM 8000)
curl -s http://localhost:30000/v1/models || curl -s http://localhost:8000/v1/models
```
> 모델 최초 다운로드는 수 분~수십 분(32B-AWQ ~20GB). `/data/hf_cache`에 받아두면 다음부터 즉시.

### Step 5. Neo4j 기동 (CPU) + 덤프 로드 — background
GPU LIVE 워크로드는 docker가 없을 수 있으므로 **Neo4j 바이너리**를 쓴다(데이터는 스토리지에).
```bash
cd /data
# Neo4j Community 5.26 (덤프 버전과 호환) 다운로드·해제
wget -q https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz
tar xzf neo4j-community-5.26.0-unix.tar.gz
export NEO4J_HOME=/data/neo4j-community-5.26.0
# 데이터 디렉토리를 스토리지에 고정 (보존)
sed -i 's|#server.directories.data=.*|server.directories.data=/data/neo4j_data|' $NEO4J_HOME/conf/neo4j.conf

# (있으면) 덤프 적재 — Community는 DB명 neo4j 고정
$NEO4J_HOME/bin/neo4j-admin database load neo4j \
  --from-path=/data --overwrite-destination=true

# 비밀번호 설정 후 기동
$NEO4J_HOME/bin/neo4j-admin dbms set-initial-password changeme123
$NEO4J_HOME/bin/neo4j start
# 준비 확인
until cypher-shell -a bolt://localhost:7687 -u neo4j -p changeme123 'RETURN 1' 2>/dev/null; do sleep 3; done
```
> 덤프가 없고 처음부터 적재하려면 `scripts/neo4j_load/run_all.py`(Day0 파이프라인) 실행.

### Step 6. 시뮬 실행
```bash
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=changeme123
export LLM_BASE_URL=http://localhost:30000/v1   # LLM 서버 포트에 맞춤
export LLM_MODE=qwen8b                          # 또는 qwen32b
export SIM_OUTPUT_DIR=/data/sim_output          # 산출물도 스토리지에

# (백테스트면) 정책 사전점검 후 적재
python scripts/sim/policy_preflight.py data/neo4j_load/policies/P010.json
python scripts/neo4j_load/09_coupon_eligibility.py       # 쿠폰 사용처 백필
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P010.json

# 본 실행 (workers는 GPU/처리량 보고 조정 — 32B+A100이면 32 근처)
python scripts/sim/run_simulation.py --start 2026-05-25 --days 3 --workers 8
```

### Step 7. 결과 보존
- 산출물이 `SIM_OUTPUT_DIR=/data/sim_output`(스토리지)에 쌓이는지 확인.
- 워크로드를 내리기 전, **필요한 것이 전부 `/data`(스토리지)에 있는지** 재확인 — 워크로드 로컬(`/home/ubuntu` 등 마운트 밖)은 삭제 시 사라진다.

---

## 3. 자주 겪는 함정
| 증상 | 원인·해결 |
|---|---|
| 워크로드 삭제 후 데이터 없음 | 마운트 밖(`/home/ubuntu`)에 저장 → **항상 `/data`(스토리지)에** |
| 모델 매번 재다운로드 | `HF_HOME`을 스토리지로 (`export HF_HOME=/data/hf_cache`) |
| 시뮬이 LLM 연결 실패 | `LLM_BASE_URL` 포트 불일치 — `curl .../v1/models`로 실제 포트 확인(30000 vs 8000) |
| vLLM OOM / shm 부족 | 워크로드 생성 시 **공유 메모리↑**, 또는 8B·AWQ 모델로 |
| Neo4j 덤프 load 실패(버전) | 덤프가 만든 버전 ≥ 5.26이면 Community **5.26**으로 |
| GPU가 안 잡힘 | 기본 이미지가 CUDA 포함(NVIDIA PyTorch)인지, `nvidia-smi` 확인 |

---

## 4. 방법 B — 사용자 이미지 (반복 운영용, 선택)

매번 설치가 번거로우면 **우리 스택을 미리 담은 이미지**를 만들어 프로젝트 레지스트리에 push한다.
GPU LIVE 사용자 이미지는 **SSH/Jupyter/supervisor/tini 연동 규약**(가이드 3.8 템플릿)을 그대로 두고,
`② 추가 패키지` 블록만 우리 것으로 채우면 된다.

```dockerfile
# syntax=docker/dockerfile:1
# ① 베이스 (CUDA 포함 — vLLM용)
ARG IMAGE_TAG=24.10-py3
FROM nvcr.io/nvidia/pytorch:${IMAGE_TAG}
# ... (가이드 3.8의 규약 블록 그대로: apt, jupyter, ubuntu 계정, sshd, supervisor, entrypoint, tini) ...
# ② 추가로 우리 런타임
RUN python3 -m pip install \
    "neo4j>=5.20,<7" "openai>=1.54,<2" "pydantic>=2.7,<3" \
    "pyyaml>=6.0" "openpyxl>=3.1" "requests>=2.31" \
    vllm            # LLM 서빙 (또는 sglang)
# 코드는 이미지에 굽지 말고 스토리지에서 git clone 권장 (업데이트 유연)
```
빌드·push (콘솔 '사용자 이미지' 탭의 Push 가이드에서 레지스트리 주소·프로젝트 ID·로봇계정 확인):
```bash
docker login registry.{도메인}
docker buildx build --platform linux/amd64 \
  --tag registry.{도메인}/{프로젝트ID}/kw26-sim:latest --push .
```
> 우리가 만든 GHCR 이미지(`ghcr.io/wikriwiki/kw_26_final_project`)는 **CPU 전용 + GPU LIVE 규약 없음**이라
> 이 플랫폼엔 그대로 못 올린다. GPU LIVE용은 위 규약 이미지를 별도로 만들어야 한다.

---

## 5. 빠른 체크리스트
```
□ 스토리지 생성 → BOUND 확인 → 워크로드에 /data 마운트
□ GPU 워크로드(인터랙티브, NVIDIA PyTorch 기본이미지) 생성 → 실행 중
□ SSH/Jupyter 접속, nvidia-smi로 GPU 확인
□ /data 에서 git clone + 덤프 SFTP 업로드
□ HF_HOME=/data/hf_cache → LLM 서버 기동(scripts/serve) → /v1/models 응답
□ Neo4j 바이너리 기동(data=/data/neo4j_data) + 덤프 load → cypher RETURN 1
□ 환경변수(NEO4J_*, LLM_BASE_URL, SIM_OUTPUT_DIR=/data/...) → 시뮬 실행
□ 종료 전: 결과가 전부 /data(스토리지)에 있는지 재확인
```
```

---

## 참고 — 우리 프로젝트 실행 인터페이스 (환경변수)
| 변수 | 기본 | 의미 |
|---|---|---|
| `NEO4J_URI` | bolt://neo4j:7687 | GPU LIVE에선 `bolt://localhost:7687` |
| `NEO4J_PASSWORD` | changeme123 | Neo4j 비밀번호 |
| `LLM_BASE_URL` | http://…:30000/v1 | LLM 서버 (SGLang 30000 / vLLM 8000) |
| `LLM_MODE` | qwen8b | qwen8b / qwen32b / qwen35_9b_awq 등 |
| `SIM_OUTPUT_DIR` | ~/sim_output | **스토리지 경로로 지정** |
| LLM 기동 | `scripts/serve/run_vllm_qwen3_8b_awq.sh` (8B) · `serve_qwen32b.sh` (32B) | |
