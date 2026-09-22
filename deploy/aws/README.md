# EC2 읽기 전용 배포

이 구성은 이미 존재하는 시뮬레이션 결과만 제공한다. 컨테이너와 데이터 볼륨은
읽기 전용이고, FastAPI는 `GET`, `HEAD`, `OPTIONS` 이외의 `/api/*` 요청을 403으로
거부한다. 허가된 사용자 IP는 EC2 보안 그룹의 `8000/tcp` 인바운드 규칙으로 제한한다.

## EC2 보안 그룹

- 인바운드 `22/tcp`: 운영자와 열람자의 고정 공인 IP만 허용한다.
- 인바운드 `8000/tcp`: 허가된 사용자의 공인 IP(`/32`)만 허용한다.
- 인바운드 `80`, `443`: 열지 않는다.
- 사용자별 SSH 키를 발급하고 공유 키를 사용하지 않는다.

## 서버 디렉터리

```bash
sudo mkdir -p /srv/policy-sim/{data,output/report,viz,agent-console}
sudo chown -R "$USER":"$USER" /srv/policy-sim
```

`data`에는 `out_BASE`, `out_FINAL`, `rescue/out_BASE7500`, `logs_scripts`를 둔다.
`output/report`에는 기존 `output/sim/report`의 보고서를, `viz`에는
`sim_standalone.html`을 둔다. `agent-console`에는
`web/ui/public/agent-console`의 조회용 JSON을 둔다. 이 대용량 파일들은 Git과
컨테이너 이미지에 넣지 않고 EC2에 별도로 전송한다. 애플리케이션은 파일을 수정하지 않는다.

## 실행

저장소 루트에서:

```bash
cp deploy/aws/.env.example deploy/aws/.env
docker compose --env-file deploy/aws/.env \
  -f deploy/aws/docker-compose.readonly.yml up -d --build
docker compose --env-file deploy/aws/.env \
  -f deploy/aws/docker-compose.readonly.yml ps
curl http://127.0.0.1:8000/api/health
```

쓰기 차단도 확인한다.

```bash
curl -i -X POST http://127.0.0.1:8000/api/runner/start \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"BASE","policy_id":"P010"}'
# HTTP 403 이어야 한다.
```

## 사용자 접속

브라우저에서 다음 주소를 연다.

```text
http://EC2_PUBLIC_DNS:8000
```

이 주소는 HTTPS가 아니므로 비밀번호나 민감한 입력을 받지 않는다. 서버는 읽기
전용이며, 보안 그룹에 등록한 IP 이외에는 연결할 수 없어야 한다.

## 갱신과 로그

```bash
git pull --ff-only
docker compose --env-file deploy/aws/.env \
  -f deploy/aws/docker-compose.readonly.yml up -d --build
docker compose --env-file deploy/aws/.env \
  -f deploy/aws/docker-compose.readonly.yml logs --tail=200 console
```
