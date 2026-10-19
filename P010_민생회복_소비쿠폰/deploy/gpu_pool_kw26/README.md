# KW26 GPU 풀 — Colab 은 GPU 만 빌려준다

시뮬레이션·Neo4j·결제 기록·기억·백업은 **우리 A100 서버에만** 있다. Colab 은 언어모델 계산만 하고,
DB·서버 셸·저장소 권한이 없다. doinggyu 브랜치(`claude/colab-gpu-integration-999c0b`)의 프록시·노트북을 우리 서버에 맞게 고쳤다.

```
[우리 A100 서버]  시뮬레이션 (새 실행: LLM_BASE_URL=http://127.0.0.1:30000/v1)
                     │
                     ▼
               중계 프로그램 127.0.0.1:30000  (/data/gpu_pool, tmux gpupool, keeper.sh)
                 ├─ A100 127.0.0.1:8000   ← P013 본런이 직접 쓰는 중. 중계는 대체 경로로만, 동시 4개까지
                 └─ Colab 계정 N : 127.0.0.1:180N1, 180N2  ◀═ SSH 역방향 터널 (계정 1~6)
```

## Colab 띄우기 (계정마다)
1. `kw26_colab_sglang_worker.ipynb` 를 Colab 에서 연다(구글 드라이브의 이 폴더에서 바로 열 수 있다).
2. 런타임 유형 **G4** (없으면 A100/H100, 40GB 이상).
3. 보안 비밀(Secrets) 두 개 + 노트북 액세스 켜기:
   - `KW26_POOL_ACCOUNT` = 그 계정 번호 1~6 (계정마다 다르게)
   - `KW26_POOL_KEY` = 키 파일 `C:\Users\Administrator\.ssh\kw26_colab_pool_<번호>` 내용 전체
   - **키는 메신저·GitHub·채팅으로 보내지 않는다.** 파일에서 Secrets 로 바로 옮긴다.
4. 모두 실행 → 마지막 셀은 멈추지 않는다. 24시간마다(또는 끊기면) 모두 실행을 다시 누른다.

## 전용 키가 할 수 있는 것 (서버 authorized_keys, 2026-10-05 시험함)
| 시도 | 결과 |
|---|---|
| 서버 셸 열기 | 거부 (`nologin`) |
| 자기 계정 포트 180N1·180N2 열기 | 허용 |
| 다른 계정 포트 열기 | 거부 (`remote port forwarding failed`) |
| 서버 안의 Neo4j(7687) 등으로 들어가기 | 거부 (`administratively prohibited`) |

## 중계 프로그램 설정 (`config.json`)
- `local_routing: false` — Colab 이 하나라도 받을 수 있으면 A100 으로 보내지 않는다(P013 이 A100 을 혼자 쓰는 동안).
- `local_max_inflight: 4` — Colab 이 전부 없을 때·Colab 실패를 A100 으로 넘길 때 동시 4개까지만.
- **P013 이 끝나면** `local_routing: true`, `local_max_inflight` 삭제 후 중계를 다시 띄운다(keeper 가 2초 뒤 재시작).
- 동일성 기준은 A100 의 `/get_server_info` 다. **A100 SGLang 을 다시 띄우면 `random_seed` 가 바뀌므로**
  `build_colab_notebook.py` 의 `SEED` 를 새 값으로 고쳐 노트북을 다시 만들고 Colab 도 다시 띄운다.

## 시험
`python -m pytest tests/unit/deploy/test_gpu_pool_kw26_proxy.py -q` — 원래 10개 + 추가 4개(A100 미사용·동시 상한·대체 상한·기본값 유지).
추가 3개는 원래 프록시에서 실패한다(확인함).
