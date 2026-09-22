# CI/CD — main push → 이미지 빌드 → GPU 서버 자동 배포

흐름: **main에 push → GitHub Actions가 이미지 빌드 → GHCR에 push → 게이트 경유 SSH로 GPU 서버가 새 이미지 pull & 재기동.**
한 번만 아래 세팅을 하면, 이후엔 코드 push만으로 끝납니다 (서버에서 환경설정 다시 X).

---

## 한 번만 하는 세팅 (3가지)

### ① GHCR 이미지 받기 권한
빌드된 이미지는 `ghcr.io/wikriwiki/kw_26_final_project` 에 올라갑니다. GPU 서버가 이걸 pull하려면 둘 중 하나:
- **(간단) 패키지를 public 으로**: GitHub → 레포 → Packages → 해당 패키지 → Package settings → Change visibility → Public
- **(비공개 유지) 서버에서 1회 로그인**: GPU 서버에서
  ```bash
  echo <READ_PAT> | docker login ghcr.io -u <github-id> --password-stdin
  ```
  (READ_PAT = `read:packages` 권한 PAT)

### ② GitHub Secrets 등록 (레포 → Settings → Secrets and variables → Actions)
| 시크릿 | 값 예시 | 설명 |
|---|---|---|
| `SSH_PRIVATE_KEY` | `-----BEGIN OPENSSH...` | 배포용 개인키 (아래 ③에서 생성) |
| `GATE_HOST` | `ubuntu@<게이트-공인IP>` | 점프(요새) 서버 |
| `GPU_HOST` | `ubuntu@<GPU-사설IP>` | 실제 배포 대상 (게이트에서 보이는 주소) |
| `APP_DIR` | `/home/ubuntu/kw_26_final_project` | GPU 서버의 레포 경로 |

### ③ 배포 키 생성 + 서버 1회 부트스트랩
로컬에서:
```bash
ssh-keygen -t ed25519 -f deploy_key -N ""        # deploy_key(개인키) / deploy_key.pub(공개키)
# deploy_key 내용을 위 SSH_PRIVATE_KEY 시크릿에 붙여넣기
# deploy_key.pub 를 게이트·GPU 서버의 ~/.ssh/authorized_keys 에 추가
```
GPU 서버에서 (게이트 경유 접속 후) 한 번만:
```bash
# 레포 클론 (LFS 데이터는 이미지에 있으니 skip → 가볍게)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/wikriwiki/kw_26_final_project.git
cd kw_26_final_project

# 서버 설정 .env (배포마다 안 바뀜 — 한 번만)
cat > .env <<'EOF'
NEO4J_PASSWORD=원하는비번
LLM_BASE_URL=http://host.docker.internal:30000/v1   # 같은 서버의 vLLM/SGLang
LLM_MODE=qwen8b
EOF

# Neo4j 기동 + Day0 그래프 적재 (최초 1회)
docker compose -f docker-compose.prod.yml up -d neo4j
docker compose -f docker-compose.prod.yml run --rm app scripts/neo4j_load/run_all.py
```

---

## 이후 운영 (자동)
- main에 push → Actions가 빌드·배포까지 자동.
- 수동 실행: 레포 → Actions → **build-and-deploy** → Run workflow.

## 시뮬 실행 (코드 갱신 후, GPU 서버에서)
```bash
cd $APP_DIR
docker compose -f docker-compose.prod.yml run --rm app \
  scripts/sim/run_simulation.py --start 2026-05-25 --days 3 --workers 8
# 결과 → ./sim_output/
```
> LLM 서버(vLLM/SGLang)는 GPU 서버에서 `scripts/serve/serve_qwen32b.sh` 등으로 별도 기동.

## 참고
- 시크릿(`SSH_PRIVATE_KEY`)이 없으면 배포 단계는 자동 skip되고 **빌드만** 됩니다 (워크플로는 green).
- 롤백: GHCR의 이전 `:<git-sha>` 태그로 `docker compose` 이미지 지정 후 `up -d`.
