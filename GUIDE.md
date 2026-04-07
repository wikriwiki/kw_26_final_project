# Agent Generation & Validation Guide

## 사전 준비

- RTX 5090 + WSL2 Ubuntu + conda `vllm` 환경
- `Qwen/Qwen3-32B-AWQ` 모델 (HuggingFace 캐시 완료)
- Windows에 `pip install openai tqdm` 설치 완료

## 1. 통계 JSON 생성

G 드라이브에서 (Windows 터미널 또는 Claude Code):

```bash
cd "G:\내 드라이브\Kw\final_project"
python analyze_stats.py
```

- `output/stats/` 에 9개 JSON 생성
- `TARGET_AGENTS` (analyze_stats.py 31번줄)로 총 에이전트 수 조절 (현재 15000)

## 2. C 드라이브로 복사

WSL에서 G 드라이브 접근 불가 -> C 드라이브에 복사 후 실행:

```powershell
# PowerShell에서
Copy-Item -Path "G:\내 드라이브\Kw\final_project\output\stats\*.json" -Destination "C:\Users\Administrator\project\final_project\output\stats\" -Force
Copy-Item -Path "G:\내 드라이브\Kw\final_project\generate_agents.py" -Destination "C:\Users\Administrator\project\final_project\" -Force
```

## 3. vLLM 서버 시작

WSL 터미널 탭 1:

```bash
wsl
conda activate vllm
vllm serve Qwen/Qwen3-32B-AWQ \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000 \
  --trust-remote-code
```

`Uvicorn running on http://0.0.0.0:8000` 이 뜨면 준비 완료. 이 탭은 그대로 유지.

## 4. 에이전트 생성

WSL 터미널 탭 2:

```bash
wsl
conda activate vllm
cd /mnt/c/Users/Administrator/project/final_project

# 시범 실행 (5그룹만)
PYTHONIOENCODING=utf-8 python generate_agents.py --limit 5

# 전체 실행
PYTHONIOENCODING=utf-8 python generate_agents.py --max-concurrent 8

# 중단 후 이어서
PYTHONIOENCODING=utf-8 python generate_agents.py --max-concurrent 8 --resume
```

- 결과: `output/agents/agents_final.json`
- 중간 저장: `output/agents/partial/batch_XXXX.json`
- 15,000명 기준 약 3~6시간 소요

## 5. 검증

```bash
# 검증 1: 에이전트 스키마 + 통계 JSON 대비 분포 검증
PYTHONIOENCODING=utf-8 python validate_agents.py

# 검증 2: 에이전트 vs raw 데이터 (telecom_29.csv 등) 평균/분산 비교
PYTHONIOENCODING=utf-8 python validate_vs_raw.py

# 문제 에이전트 제거 후 저장
PYTHONIOENCODING=utf-8 python validate_agents.py --fix
```

### validate_agents.py 검증 항목
- 그룹별 할당 수량 일치
- spending_level / mobility_level 분포
- 성별 / 연령대 / 자치구 분포
- 업종별 소비비율
- 소득 수준 / 직업 다양성
- 개별 에이전트 스키마 및 논리 검증

### validate_vs_raw.py 검증 항목
- 텔레콤 지표 (출근시간, 배달일수, 이동거리 등) raw 평균 vs 에이전트 평균
- 성별x연령대 인구 분포 비교
- 자치구 분포 비교
- aggregate stats (mean/std) vs 에이전트 분포

## 6. 결과 복사 (G 드라이브로)

```powershell
# PowerShell에서
Copy-Item -Path "C:\Users\Administrator\project\final_project\output\agents\*" -Destination "G:\내 드라이브\Kw\final_project\output\agents\" -Recurse -Force
```

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--max-concurrent` | 8 | 동시 LLM 요청 수 |
| `--limit N` | 0 (전체) | N개 그룹만 처리 |
| `--resume` | off | 중단 지점부터 재개 |
| `--vllm-url` | localhost:8000/v1 | vLLM API 주소 |
| `TARGET_AGENTS` | 15000 | 총 에이전트 수 (analyze_stats.py) |
