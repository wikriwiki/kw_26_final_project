#!/usr/bin/env python3
"""낮은 우선순위 관문 (2026-10-08) — 뒤에 붙는 런이 앞선 런의 GPU 몫을 빼앗지 않게 한다.

사용자: "빈틈이 생기는 단계에 (다음 정책을) 돌리도록 해." 지금 돌고 있는 GPU 풀 중계(30100)는 그대로 두고,
그 앞에 이 관문(30101)을 세운다. 뒤에 붙는 런만 30101 을 쓴다.

생성 요청 하나를 넘기기 전에 중계의 /pool/status 를 읽어
    다른 런이 쓰는 자리 = 전체 inflight − 이 관문이 넘긴 inflight
    용량 = 건강한 일꾼 수 × 일꾼당 자리(기본 64, 일꾼 하나가 포화하는 동시 수)
다른 런 + 이 관문 < 용량 일 때만 넘긴다. 아니면 기다린다(앞선 런이 우선). 요청·응답 바이트는 그대로 전달한다.
/v1/models · /pool/status 같은 조회는 바로 넘긴다. 표준 라이브러리만 쓴다.
"""
import argparse
import http.client
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers',
       'transfer-encoding', 'upgrade', 'host', 'content-length'}


class Gate:
    def __init__(self, upstream_host, upstream_port, per_backend, log_path):
        self.host, self.port, self.per = upstream_host, upstream_port, per_backend
        self.lock = threading.Lock()
        self.own = 0
        self.status = None
        self.status_at = 0.0
        self.log_path = log_path
        self.waited = 0

    def pool(self):
        now = time.time()
        if self.status is None or now - self.status_at > 1.0:
            try:
                c = http.client.HTTPConnection(self.host, self.port, timeout=5)
                c.request('GET', '/pool/status')
                self.status = json.loads(c.getresponse().read())
                self.status_at = now
            except Exception:
                self.status = None
        return self.status

    def acquire(self):
        started = time.time()
        while True:
            st = self.pool()
            if st is not None:
                healthy = [b for b in st.get('backends', []) if b.get('healthy')]
                total = sum(int(b.get('inflight') or 0) for b in healthy)
                cap = self.per * len(healthy)
                with self.lock:
                    others = max(0, total - self.own)
                    if others + self.own < cap:
                        self.own += 1
                        self.status_at = 0.0          # 다음 판단은 새 상태로
                        return time.time() - started
            time.sleep(0.5)

    def release(self):
        with self.lock:
            self.own -= 1

    def log(self, **kw):
        kw['at_utc'] = datetime.now(timezone.utc).isoformat()
        with self.lock, open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(kw, ensure_ascii=False) + '\n')


def make_handler(gate):
    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _forward(self, body, timeout):
            c = http.client.HTTPConnection(gate.host, gate.port, timeout=timeout)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            c.request(self.command, self.path, body=body, headers=headers)
            r = c.getresponse()
            data = r.read()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return r.status

        def do_GET(self):
            try:
                self._forward(None, 30)
            except Exception as exc:
                self.send_error(502, repr(exc)[:200])

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b''
            routed = self.path.startswith('/v1/chat/completions') or self.path.startswith('/v1/completions')
            waited = gate.acquire() if routed else 0.0
            t0 = time.time()
            status = None
            try:
                status = self._forward(body, 3600)
            except Exception as exc:
                self.send_error(502, repr(exc)[:200])
            finally:
                if routed:
                    gate.release()
                    gate.log(path=self.path, waited=round(waited, 2), seconds=round(time.time() - t0, 2),
                             status=status, own_after=gate.own)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen-port', type=int, default=30101)
    ap.add_argument('--upstream-port', type=int, default=30100)
    ap.add_argument('--per-backend', type=int, default=64)
    ap.add_argument('--log', default='/data/gpu_pool/priority_gate.jsonl')
    a = ap.parse_args()
    gate = Gate('127.0.0.1', a.upstream_port, a.per_backend, a.log)
    srv = ThreadingHTTPServer(('127.0.0.1', a.listen_port), make_handler(gate))
    srv.daemon_threads = True
    srv.request_queue_size = 1024
    print(f'priority gate 127.0.0.1:{a.listen_port} -> {a.upstream_port}, per-backend {a.per_backend}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
