# GPU 추론 풀 — Vast 본 실험에 Colab GPU를 붙여 가속하는 방법

대상: `No_SmokingZone_EXP` 본 실험(노원·서초·송파 1,154명, EXAONE-4.5-33B-AWQ, SGLang).
상태: **2026-09-30 11:05 UTC부터 본 실험에 적용 중.** 상세 이력·해시·검사 결과는 [`docs/HANDOFF_GPU_POOL.md`](../../docs/HANDOFF_GPU_POOL.md).

## 한 줄 요약

실험(시뮬레이터·DB·검증·재시도 횟수·백업)은 **Vast 서버 한 곳에서만** 돌고, LLM 요청만 Vast GPU와 Colab GPU가 나눠 처리한다. Colab이 끊기면 요청이 자동으로 Vast로 돌아가고 실험은 멈추지 않는다.

## 구조

```text
[Vast 서버]                                            [Colab G4 런타임]
 simulator (workers=8) ── LLM 요청 ──▶ 프록시 :30000
   · Neo4j, 결과 저장, 검증,              ├─▶ local      127.0.0.1:8000  (Vast L40S의 SGLang)
     재시도 예산, Drive 백업              ├─▶ colab-g4-0 127.0.0.1:18001 ══ SSH 역방향 터널 ══▶ SGLang :8000
   · 전부 여기서만 수행                   └─▶ colab-g4-1 127.0.0.1:18002 ══ SSH 역방향 터널 ══▶ SGLang :8001
```

- **프록시**(`gpu_pool_proxy.py`, 표준 라이브러리만 사용)가 요청 단위로 가장 한가한 서버에 보낸다. 가중치: Vast 1, Colab 서버 각 1.5 → 동시 8요청이면 Vast 2, Colab 3+3.
- **동일 모델 보장:** 원격 서버는 `/get_server_info`의 모델·revision·context·문법 backend·seed·SGLang 버전 등이 Vast와 전부 같을 때만 요청을 받는다. 터널이 다시 붙을 때마다 재검사한다.
- **장애 시:** 원격이 거부·끊김·5xx·health 상실이면 같은 요청을 Vast로 한 번 다시 보낸다. 시뮬레이터는 요청당 응답 하나만 받으므로 중복 저장이 없고, 통신 재전송은 agent-day 재시도 예산(최초 1회+5회)을 쓰지 않는다. 뒤늦게 도착한 원격 응답은 버리고 기록만 남긴다.
- **Colab은 추론만** 한다. DB·Drive·Vast 셸 권한이 없다. 터널용 SSH 키는 Vast에서 셸이 막혀 있고 `127.0.0.1:18001~18004` 리슨만 허용된다.
- **출처 기록:** 모든 요청의 처리 서버·요청/응답 SHA256·소요 시간이 JSONL로 남는다.

## 왜 이렇게 했나

- 병목은 GPU 메모리가 아니라 **SGLang 스케줄러의 CPU 한 코어**였다(Vast: CPU 100%, GPU 약 40%). 요청을 여러 서버로 나누면 서버마다 부담이 줄어든다.
- `workers=8`은 바꿀 수 없다. 재개 시 `workers`·`server_config`·`engine_settings`·코드 해시가 달라지면 실행이 거부된다(`Resume cannot change the frozen execution`). 그래서 동시 요청 8개를 나누는 방식으로만 가속한다.
- 홀수/짝수 ID 고정 분배는 쓰지 않는다. 느린 쪽을 기다리게 되고, Colab이 끊기면 그쪽 몫이 멈춘다.
- 시뮬레이터를 두 곳에서 따로 돌려 DB를 합치는 방식은 원장·백업·계보와 맞지 않아 쓰지 않는다.

## 실측 (2026-09-30)

| 항목 | 값 |
|---|---|
| Colab GPU | RTX PRO 6000 Blackwell 97.9GB(G4), 48 vCPU, nvcc 12.8.93(Vast와 동일) |
| G4 단독(터널 경유, 입력 6.6K 토큰, 동시 6요청) | 합계 125 tok/s, 요청당 21 tok/s |
| Vast 단독(운영 중 동시 8요청) | 합계 약 72~78 tok/s, 요청당 약 9~10 tok/s |
| **본 실험 처리량** | Vast 단독 약 120~127명/시간 → **풀 적용 후 약 300명/시간(약 2.4배)** |
| 적용 후 첫 1시간 | 로컬 전환 0건, 프록시 실패 0건 |

G4에 서버를 1개→2개로 늘린 이득은 약 16%였고, 3개째는 메모리 여유가 부족해 쓰지 않는다.

## Colab 워커 실행 (세션이 끊길 때마다 반복)

1. [`colab_sglang_worker.ipynb`](colab_sglang_worker.ipynb)를 Colab에 업로드하고 **G4(또는 H100/A100, 40GB 이상)** 런타임에 연결한다.
2. 🔑 Secrets에 `VAST_GPU_POOL_KEY`(전용 제한 키 본문), `VAST_SSH_HOST`, `VAST_SSH_PORT`를 등록하고 **노트북 액세스**를 켠다. 키는 실험 담당자에게 받는다. 저장소·채팅·셀 출력에 넣지 않는다.
3. **런타임 → 모두 실행.** 첫 실행은 설치·모델 다운로드로 10~20분.
4. **마지막 셀은 계속 실행 상태로 둔다.** 실행 중인 셀이 없으면 Colab이 약 90분 뒤 유휴로 세션을 끊는다(실제로 한 번 겪었다). Pro+도 연속 실행은 최대 24시간이다.

터널이 붙으면 Vast 프록시가 자동으로 검사 후 투입한다. Vast 쪽에서 따로 할 일은 없다.

## 상태 확인 (Vast에서, 읽기 전용)

```bash
python3 /workspace/no-smoking-gpu-pool/pool_monitor.py        # 진행·프로세스·백엔드별 요청 수/평균 초·오류·디스크
curl -s http://127.0.0.1:30000/pool/status                    # 백엔드 상태
```

| 로그 | 내용 |
|---|---|
| `/workspace/no-smoking-results/integration-main-v22-1154-gpu-pool.jsonl` | 요청별 처리 서버, 요청/응답 SHA256, 전환(fallback), 투입/제외 이벤트 |
| `…-gpu-pool-routing.jsonl` | 시뮬레이터 프로세스가 시작할 때 프록시를 쓰기로 했는지(`pool`/`direct`) |
| `…-gpu-pool-switch.log` | 본 실험 교체 기록 |
| `…-resume-control.jsonl` | 복구기 이벤트 |

## 서버 구성 요소

| 위치 | 역할 |
|---|---|
| `/workspace/no-smoking-gpu-pool/` | `gpu_pool_proxy.py`, `config.json`, `keeper.sh`(죽으면 1초 뒤 재시작), `pool_monitor.py`, `switch_at_boundary.sh` |
| `/workspace/no-smoking-runtime-hotfix-v22-gpu-pool/` | 런타임 층. 저장소의 [`deploy/vast/runtime_hotfix_v22_gpu_pool/`](../vast/runtime_hotfix_v22_gpu_pool/)와 동일 |
| `/workspace/onstart.sh` | 새 복구기 실행(이전 것은 `onstart.sh.pre-gpu-pool-*`로 보존) |

frozen v22 코드는 모델 주소를 `127.0.0.1:8000`으로 고정한다. 그래서 기존 패치들과 같은 방식으로 **맨 위에 런타임 층 하나**를 얹었다.

- `sitecustomize.py`: 기존 night-progress 층을 해시 검증 후 그대로 실행하고, 프록시가 살아 있으면 시뮬레이터 프로세스의 `SGLANG_BASE_URL`만 `127.0.0.1:30000`으로 바꾼다. 생성·검증·재시도·저장 코드는 건드리지 않는다.
- `resume_runtime.py`: 기존 복구기를 해시 검증 후 그대로 불러 쓰고, 감독자 실행 시 이 층을 PYTHONPATH 맨 앞에 넣는다. 같은 잠금 파일을 쓰므로 옛 복구기와 동시에 돌 수 없다. 프록시 keeper가 죽어 있으면 다시 띄운다.
- 실행 지문(execution/source fingerprint)은 이 층을 얹어도 기존 날짜 기록과 같다. 모델 주소는 지문·`engine_settings`에 들어가지 않는다.

## 끄는 법 / 되돌리는 법

- **Colab만 빼기:** Colab 노트북을 멈추면 된다. 프록시가 제외하고 Vast로 돌린다.
- **일시 제외(세션 유지):** `curl -X POST 'http://127.0.0.1:30000/pool/drain?name=colab-g4-0'` (되돌리기는 `/pool/undrain`).
- **다음 재시작부터 프록시 미사용:** `touch /workspace/no-smoking-gpu-pool/DISABLED`. 실행 중인 시뮬레이터는 그대로이고, 새로 시작하는 프로세스부터 8000 직결.
- **완전 복귀:** 새 복구기와 감독자를 정상 종료(SIGTERM)한 뒤 `/workspace/onstart.sh.pre-gpu-pool-*`를 `/workspace/onstart.sh`로 복원해 실행. 재개는 기존 결과에서 이어진다.

## 하지 말 것

- 실행 중인 프록시를 임의로 죽이지 않는다. keeper가 1~2초 안에 다시 띄우지만 그 사이 진행 중이던 요청은 실패해 해당 에이전트의 재시도 1회가 소모된다.
- 프록시에 다른 모델·다른 버전 서버를 붙이려고 identity 검사를 풀지 않는다. 버전이 다르면 "동일 모델 실행"이 아니다.
- `workers`, frozen 코드, manifest 해시를 바꾸지 않는다. 재개가 거부된다.
- Day1부터 재실행하거나 봉인된 결과를 지우지 않는다. 모델 서버·Vast 인스턴스를 불필요하게 재시작하지 않는다.
- 비밀 키·암호를 저장소·문서·노트북 출력에 넣지 않는다.

## 새 GPU 환경에서 같은 서버를 만들 때 걸렸던 것

| 증상 | 원인과 해결 |
|---|---|
| `uv pip install`이 `transformers==5.3.0` 충돌로 실패 | Vast는 sglang 설치 뒤 transformers를 5.8.0으로 올렸다. freeze가 전체 목록이므로 `--no-deps`로 그대로 설치 |
| 서버 시작 시 `FileNotFoundError: 'ninja'` | 가중치 변환 커널을 즉석 컴파일한다. 실행 PATH에 venv `bin`(ninja)과 `/usr/local/cuda/bin`(nvcc) 추가 |
| 두 번째 서버가 `Not enough memory … mem_fraction_static` | 이 값은 GPU를 혼자 쓴다고 가정한다. 남은 메모리 기준으로 계산해 넘긴다(노트북의 `fraction_for`) |
| 터널 `Load key … error in libcrypto` | Colab Secrets가 여러 줄을 한 줄로 합친다. 노트북이 개인키 줄바꿈을 복원한다 |
| 90분 뒤 세션 종료 | 실행 중인 셀이 없었다. 마지막 상태 루프 셀을 계속 실행 |
| `SM 12.x requires CUDA >= 12.9` 경고 | Blackwell에서 나오지만 서버는 정상 동작했다(경고로만 확인) |

## 코드와 검사

| 파일 | 내용 |
|---|---|
| `gpu_pool_proxy.py` | 프록시 |
| `build_colab_notebook.py` → `colab_sglang_worker.ipynb` | 노트북 생성기와 결과물 |
| `vast_venv_sgl_freeze.txt` | Vast `venv_sgl` 전체 패키지 목록(sglang = `lkm2835/sglang@6757c9f9`, torch 2.9.1, outlines 0.1.11, Python 3.11) |
| `switch_at_boundary.sh` | 백업 검증된 날짜 경계에서 계산 프로세스만 교체하고, 검증 실패 시 자동 복귀 |
| `../../tests/unit/deploy/test_gpu_pool_proxy.py` | 가짜 SGLang으로 프록시 10개 시험 |

```bash
python -m pytest tests/unit/deploy/test_gpu_pool_proxy.py -q     # 10 passed (Windows, Python 3.13)
python deploy/gpu_pool/build_colab_notebook.py                   # 노트북 다시 생성
python deploy/vast/runtime_hotfix_v22_gpu_pool/build_manifest.py # 층 manifest·해시 다시 생성
```

## 두 번째 Vast 워커 (2026-10-05 추가)

Colab의 24시간 한도·컴퓨팅 단위 제약 없이 상시 쓰는 보조 GPU. 같은 이미지·드라이버의 L40S 인스턴스에 [`vast_worker_setup.sh`](vast_worker_setup.sh)로 같은 패키지·모델을 설치하고, [`vast_worker_keeper.sh`](vast_worker_keeper.sh)가 기존 서버와 같은 인자로 SGLang을 유지하며 역방향 터널(`127.0.0.1:18003`)을 연다. 프록시에는 `vast2-l40s`(capacity 1.0)로 등록하며, 설정 교체는 [`reload_proxy_when_idle.sh`](reload_proxy_when_idle.sh)로 **백업 중·진행 요청 0건일 때만** 프록시를 재시작한다.

- 필요했던 것: OS 패키지 `libnuma1`(없으면 `sgl_kernel` import 실패), 같은 물리 머신이면 공인 IP 대신 **내부 주소**로 터널, keeper의 잠금 fd를 자식에 넘기지 않기(`9>&-`).
- **주의:** Vast가 계정 키를 재동기화하면서 실험 서버의 `/root/.ssh/authorized_keys`를 덮어써 제한 키가 사라진 적이 있다(10-01). 원격 GPU가 `Permission denied`로 붙지 않으면 이것부터 확인한다.
- 오래된 Neo4j 폴더 정리는 [`retire_old_pairs.py`](retire_old_pairs.py): 덤프 → Drive 업로드·MD5 검증 → 삭제. PID 파일 대신 실제 JVM으로 사용 여부를 판단하고 `neo4j stop`을 쓰지 않는다.

## 아직 확인하지 못한 것

- ~~본 실험 도중 Colab 끊김 시 Vast 복귀~~ → **확인됨**: 10-01 터널 순간 끊김(6건)과 24시간 한도 종료(6건) 모두 진행 중 요청이 Vast로 넘어가 성공, 실패 0.
- 프록시 시험의 Linux/Python 3.11 실행(서버에서는 프록시 실제 동작과 재시작만 확인).
- ~~OFF/ON 전환 디스크 10GB~~ → **통과**(10-01 디스크 정리로 여유 30GB 확보 후 전환, post DB 복원·마커 검증 완료).
- 서로 다른 GPU(L40S, Blackwell)는 같은 설정이어도 출력이 비트 단위로 같다는 보장이 없다. **사용자 결정(2026-10-01): 공통 선행뿐 아니라 OFF/ON 구간에서도 두 GPU를 섞어 쓴다.** 요청별 처리 서버가 `…-gpu-pool.jsonl`에 기록되므로 필요하면 arm별 GPU 비율을 사후 점검할 수 있다(라우팅은 arm과 무관하게 부하 기준).
