"""Isolated checks for the inference-only GPU pool (no real model, no Vast)."""
import http.client
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deploy.gpu_pool_kw26 import gpu_pool_proxy as gp

IDENTITY = {'model_path': 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ',
            'served_model_name': 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ',
            'revision': '31e6a965d0661bbe4a8b895e22a77f8271772ba0', 'context_length': 16384,
            'grammar_backend': 'outlines', 'constrained_json_whitespace_pattern': '[\\n\\t ]*',
            'random_seed': 42, 'dtype': 'auto', 'version': '0.0.0.dev11420+g6757c9f90'}


class FakeSGLang:
    """Minimal SGLang stand-in: identity, health, chat with delay/failure modes."""

    def __init__(self, name, port=0, **identity):
        self.name, self.delay, self.mode = name, 0.0, 'ok'
        self.info = {**IDENTITY, 'attention_backend': 'triton', **identity}
        self.healthy = True
        self.calls, self.lock = [], threading.Lock()
        self.inflight = self.peak = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def send(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == '/health':
                    return self.send(200 if fake.healthy else 503, {})
                if self.path == '/get_server_info':
                    return self.send(200, fake.info)
                if self.path == '/v1/models':
                    return self.send(200, {'data': [{'id': fake.info['served_model_name']}]})
                self.send(404, {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                with fake.lock:
                    fake.calls.append(body)
                    fake.inflight += 1
                    fake.peak = max(fake.peak, fake.inflight)
                try:
                    if fake.mode == 'hang':
                        time.sleep(30)
                    time.sleep(fake.delay)
                    if fake.mode == '500':
                        return self.send(500, {'error': 'boom'})
                    if fake.mode == 'reset':
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.send(200, b'{"served_by":"' + fake.name.encode() + b'","echo_sha":"'
                              + gp.sha256(body).encode() + b'"}')
                finally:
                    with fake.lock:
                        fake.inflight -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f'http://127.0.0.1:{self.port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def cluster(tmp_path):
    made = []

    def build(remote_identity=None, **pool_kw):
        local = FakeSGLang('local')
        remote = FakeSGLang('remote', **(remote_identity or {}))
        config = {'local_url': local.url, 'listen_port': 0, 'log_path': str(tmp_path / 'pool.jsonl'),
                  'remotes': [{'name': 'colab', 'url': remote.url}],
                  'health_interval': 0.1, 'health_timeout': 1.0, 'unhealthy_after': 2, **pool_kw}
        pool, server = gp.serve(config)
        threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True).start()
        made.append((pool, server, local, remote))
        return pool, server.server_address[1], local, remote

    yield build
    for pool, server, local, remote in made:
        pool.stop.set()
        server.shutdown()
        server.server_close()
        local.close()
        remote.close()


def post(port, body, timeout=20):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    conn.request('POST', '/v1/chat/completions', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    data = resp.read()
    backend = resp.getheader('X-GPU-Pool-Backend')
    conn.close()
    return resp.status, data, backend


def get(port, path):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', path)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data)


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def events(pool):
    return [json.loads(l) for l in open(pool.log_path, encoding='utf-8')]


def test_remote_admitted_only_with_matching_identity_and_bytes_forwarded(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    body = json.dumps({'model': IDENTITY['served_model_name'], 'seed': 7,
                       'messages': [{'role': 'user', 'content': '한글 ✓'}]}).encode()
    status, data, _ = post(port, body)
    assert status == 200
    assert json.loads(data)['echo_sha'] == gp.sha256(body)       # request bytes untouched
    assert (local.calls + remote.calls) == [body]                  # exactly one delivery
    record = [e for e in events(pool) if e['event'] == 'request'][0]
    assert record['request_sha256'] == gp.sha256(body)
    assert record['response_sha256'] == gp.sha256(data)             # response bytes untouched
    assert record['backend_identity_sha256'] == pool.local.identity_sha256


def test_identity_mismatch_never_receives_work(cluster):
    pool, port, local, remote = cluster(remote_identity={'grammar_backend': 'xgrammar'})
    assert wait_for(lambda: any(e['event'] == 'remote_rejected' for e in events(pool)))
    for i in range(6):
        assert post(port, json.dumps({'i': i}).encode())[2] == 'local'
    assert remote.calls == []
    rejected = [e for e in events(pool) if e['event'] == 'remote_rejected'][0]
    assert 'grammar_backend' in rejected['error']


def test_concurrent_requests_split_across_gpus(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    local.delay = remote.delay = 1.0
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda i: post(port, json.dumps({'i': i}).encode()), range(8)))
    served = [r[2] for r in results]
    # Least-loaded routing: both GPUs work concurrently, neither takes all 8.
    assert 3 <= served.count('colab') <= 5 and served.count('local') + served.count('colab') == 8
    assert local.peak <= 5 and remote.peak <= 5


def test_remote_death_mid_request_falls_back_once_without_duplicate_response(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    remote.mode = 'hang'
    local.delay = 0.3
    with ThreadPoolExecutor(2) as ex:
        future_a = ex.submit(post, port, b'{"a":1}')
        time.sleep(0.2)
        future_b = ex.submit(post, port, b'{"b":2}')
        time.sleep(0.2)
        remote.healthy = False                 # Colab disconnect as seen by /health
        results = [future_a.result(timeout=15), future_b.result(timeout=15)]
    assert all(r[0] == 200 and r[2] == 'local' for r in results)
    assert sorted(local.calls) == [b'{"a":1}', b'{"b":2}']
    removed = [e for e in events(pool) if e['event'] == 'remote_removed']
    assert removed and removed[0]['aborted_inflight'] >= 1
    reqs = [e for e in events(pool) if e['event'] == 'request']
    assert len(reqs) == 2 and any(e['fallback_from'] == 'colab' for e in reqs)


@pytest.mark.parametrize('mode', ['500', 'reset'])
def test_remote_error_is_transport_fallback(cluster, mode):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    remote.mode = mode
    results = [post(port, json.dumps({'i': i}).encode()) for i in range(3)]
    assert all(r[0] == 200 for r in results)
    assert all(json.loads(r[1])['served_by'] == 'local' for r in results)


def test_refused_remote_falls_back_and_is_readmitted_after_reverification(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    remote_port = remote.port
    remote.close()                              # tunnel gone: connection refused
    assert post(port, b'{"x":1}')[2] == 'local'
    assert wait_for(lambda: not pool.remotes[0].healthy)
    # A new session re-binds the same tunnel port; it must pass identity again.
    replacement = FakeSGLang('remote2', port=remote_port)
    try:
        assert wait_for(lambda: pool.remotes[0].healthy)
        admitted = [e for e in events(pool) if e['event'] == 'remote_admitted']
        assert len(admitted) == 2
        assert post(port, b'{"y":1}')[2] in {'local', 'colab'}
    finally:
        replacement.close()


def test_drain_stops_new_remote_work_and_status_reports(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('POST', '/pool/drain?name=colab')
    assert conn.getresponse().status == 200
    conn.close()
    for i in range(4):
        assert post(port, json.dumps({'i': i}).encode())[2] == 'local'
    status, body = get(port, '/pool/status')
    colab = [b for b in body['backends'] if b['name'] == 'colab'][0]
    assert colab['draining'] and colab['healthy'] and remote.calls == []


def test_non_generation_paths_always_use_local(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].healthy)
    status, body = get(port, '/v1/models')
    assert status == 200 and body['data'][0]['id'] == IDENTITY['served_model_name']
    assert get(port, '/get_server_info')[1]['attention_backend'] == 'triton'


def test_local_failure_is_reported_not_hidden(cluster):
    pool, port, local, remote = cluster()
    pool.remotes[0].draining = True
    local.mode = 'reset'
    status, data, _ = post(port, b'{"x":1}')
    assert status == 502
    assert any(e['event'] == 'request_failed' for e in events(pool))


# ---- 우리 서버 추가 설정 (2026-10-05) ----

def test_local_routing_off_sends_nothing_to_local_while_a_remote_is_eligible(cluster):
    pool, port, local, remote = cluster(local_routing=False)
    assert wait_for(lambda: pool.remotes[0].eligible())
    remote.delay = 0.2
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda i: post(port, json.dumps({'n': i}).encode()), range(6)))
    assert all(r[0] == 200 for r in results)
    assert len(remote.calls) == 6 and len(local.calls) == 0


def test_local_cap_holds_when_every_remote_is_gone(cluster):
    pool, port, local, remote = cluster(local_routing=False, local_max_inflight=2)
    remote.healthy = False
    assert wait_for(lambda: not pool.remotes[0].eligible())
    local.delay = 0.3
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda i: post(port, json.dumps({'n': i}).encode()), range(6)))
    assert all(r[0] == 200 for r in results)
    assert len(local.calls) == 6 and local.peak <= 2


def test_fallback_after_remote_failure_respects_local_cap(cluster):
    pool, port, local, remote = cluster(local_routing=False, local_max_inflight=1)
    assert wait_for(lambda: pool.remotes[0].eligible())
    remote.mode = 'reset'
    local.delay = 0.2
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda i: post(port, json.dumps({'n': i}).encode()), range(4)))
    assert all(r[0] == 200 for r in results)
    assert local.peak <= 1


def test_default_config_still_uses_local_like_before(cluster):
    pool, port, local, remote = cluster()
    assert wait_for(lambda: pool.remotes[0].eligible())
    remote.delay = local.delay = 0.3
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda i: post(port, json.dumps({'n': i}).encode()), range(4)))
    assert len(local.calls) >= 1 and len(remote.calls) >= 1
