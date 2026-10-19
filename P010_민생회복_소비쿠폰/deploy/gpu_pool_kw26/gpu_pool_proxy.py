#!/usr/bin/env python3
"""Inference-only GPU pool in front of the Vast SGLang server.

The single simulator on Vast keeps every DB write, ledger, validation, retry
budget and backup. This proxy only decides which *identical* model server
answers one HTTP request:

* ``local``  - the existing Vast SGLang server (always eligible, the fallback).
* remotes    - extra SGLang servers reached through reverse SSH tunnels that
               listen on Vast loopback ports (Colab, a second Vast box, ...).

A remote is admitted only while ``/get_server_info`` matches the local server
on every generation-relevant field and ``/health`` answers. When a remote
fails (refused, reset, timeout, 5xx, health loss) the same request bytes are
sent to the local server once. The simulator sees exactly one response per
request, so there is no late or duplicate delivery and no retry budget is
spent by a transport failure. Request and response bodies are forwarded
byte-for-byte; provenance goes to a JSONL log, never into the response body.

Standard library only (Python 3.11+).
"""

import argparse
import hashlib
import http.client
import json
import os
import socket
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Fields that change what the model generates or how the prompt is tokenized.
# Scheduling/memory knobs (attention backend, mem fraction, cuda graphs) may
# differ between GPU types; they are logged at admission but not required.
IDENTITY_FIELDS = (
    'model_path', 'served_model_name', 'revision', 'context_length',
    'grammar_backend', 'constrained_json_whitespace_pattern',
    'constrained_json_disable_any_whitespace', 'random_seed', 'dtype',
    'version', 'quantization', 'kv_cache_dtype', 'trust_remote_code',
    'chat_template', 'tokenizer_path', 'tokenizer_mode', 'sampling_defaults',
    'preferred_sampling_params', 'json_model_override_args', 'reasoning_parser',
    'skip_tokenizer_init', 'is_embedding', 'speculative_algorithm',
    'enable_deterministic_inference',
)
RUNTIME_FIELDS = ('attention_backend', 'mem_fraction_static', 'max_running_requests',
                  'tp_size', 'device', 'max_total_num_tokens', 'cuda_graph_max_bs')
ROUTED_PATHS = ('/v1/chat/completions', '/v1/completions')
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailers', 'transfer-encoding', 'upgrade', 'host', 'content-length'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def identity_of(info):
    return {k: info.get(k, '<absent>') for k in IDENTITY_FIELDS}


def identity_digest(identity):
    return sha256(json.dumps(identity, sort_keys=True).encode())


class Backend:
    def __init__(self, name, url, capacity=1.0, local=False):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'http' or not parsed.hostname or not parsed.port:
            raise ValueError(f'backend {name}: need http://host:port, got {url!r}')
        if float(capacity) <= 0:
            raise ValueError(f'backend {name}: capacity must be positive')
        self.name, self.url, self.local = name, url, local
        self.host, self.port = parsed.hostname, parsed.port
        self.capacity = float(capacity)
        self.healthy = local        # remotes stay excluded until verified
        self.draining = False
        self.inflight = 0
        self.failures = 0           # consecutive health failures
        self.identity_sha256 = None
        self.identity_error = None
        self.served = self.errors = self.fallbacks = 0
        self.conns = set()

    def eligible(self):
        return self.healthy and not self.draining

    def snapshot(self):
        return {'name': self.name, 'url': self.url, 'local': self.local,
                'capacity': self.capacity, 'healthy': self.healthy,
                'draining': self.draining, 'inflight': self.inflight,
                'served': self.served, 'errors': self.errors,
                'fallbacks_from_here': self.fallbacks,
                'identity_sha256': self.identity_sha256,
                'identity_error': self.identity_error}


class Pool:
    def __init__(self, local_url, remotes, log_path, *, health_interval=5.0,
                 health_timeout=4.0, unhealthy_after=3, remote_timeout=420.0,
                 local_timeout=900.0, connect_timeout=5.0, local_routing=True,
                 local_max_inflight=None):
        self.lock = threading.Lock()
        self.local_free = threading.Condition(self.lock)
        # 우리 서버 추가(2026-10-05): A100 을 다른 실행이 쓰는 동안 이 풀은 원격에만 일을 준다.
        self.local_routing = bool(local_routing)
        self.local_max_inflight = None if local_max_inflight in (None, 0, '') else int(local_max_inflight)
        self.local = Backend('local', local_url, local=True)
        self.remotes = [Backend(r['name'], r['url'], r.get('capacity', 1.0)) for r in remotes]
        names = [b.name for b in self.all()]
        if len(set(names)) != len(names):
            raise ValueError('backend names must be unique')
        self.log_path = log_path
        self.log_lock = threading.Lock()
        self.health_interval, self.health_timeout = health_interval, health_timeout
        self.unhealthy_after = unhealthy_after
        self.remote_timeout, self.local_timeout = remote_timeout, local_timeout
        self.connect_timeout = connect_timeout
        self.reference = None       # local identity: the only accepted identity
        self.stop = threading.Event()

    def all(self):
        return [self.local, *self.remotes]

    # ---------- provenance ----------
    def log(self, **record):
        # Provenance is best effort: a full disk must never fail an inference request.
        try:
            line = json.dumps({'at_utc': utc(), **record}, ensure_ascii=False, sort_keys=True, default=str)
            with self.log_lock:
                with open(self.log_path, 'a', encoding='utf-8') as out:
                    out.write(line + '\n')
        except Exception:
            pass

    # ---------- backend HTTP ----------
    def fetch_json(self, backend, path, timeout):
        conn = http.client.HTTPConnection(backend.host, backend.port, timeout=timeout)
        try:
            conn.request('GET', path)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f'{path} -> HTTP {resp.status}')
            return json.loads(body)
        finally:
            conn.close()

    def load_reference(self):
        info = self.fetch_json(self.local, '/get_server_info', self.health_timeout)
        self.reference = identity_of(info)
        self.local.identity_sha256 = identity_digest(self.reference)
        self.log(event='reference_loaded', identity=self.reference,
                 identity_sha256=self.local.identity_sha256,
                 runtime={k: info.get(k) for k in RUNTIME_FIELDS})

    def verify_remote(self, backend):
        info = self.fetch_json(backend, '/get_server_info', self.health_timeout)
        identity = identity_of(info)
        if identity != self.reference:
            diff = {k: {'local': self.reference[k], 'remote': identity[k]}
                    for k in IDENTITY_FIELDS if identity[k] != self.reference[k]}
            raise ValueError('identity mismatch: ' + json.dumps(diff, ensure_ascii=False)[:600])
        models = self.fetch_json(backend, '/v1/models', self.health_timeout)
        ids = [m.get('id') for m in models.get('data', [])]
        if self.reference['served_model_name'] not in ids:
            raise ValueError(f'served model missing: {ids}')
        return identity_digest(identity), {k: info.get(k) for k in RUNTIME_FIELDS}

    def health_ok(self, backend):
        conn = http.client.HTTPConnection(backend.host, backend.port, timeout=self.health_timeout)
        try:
            conn.request('GET', '/health')
            resp = conn.getresponse()
            resp.read()
            return resp.status == 200
        except Exception:
            return False
        finally:
            conn.close()

    # ---------- health loop ----------
    def check_remote(self, backend):
        if backend.healthy:
            if self.health_ok(backend):
                backend.failures = 0
                return
            backend.failures += 1
            if backend.failures >= self.unhealthy_after:
                self.mark_unhealthy(backend, 'health_lost')
            return
        # Not admitted: re-verify the full identity before every admission,
        # because a reconnected tunnel may point at a different server.
        if not self.health_ok(backend):
            return
        try:
            digest, runtime = self.verify_remote(backend)
        except Exception as exc:
            if backend.identity_error != str(exc):
                backend.identity_error = str(exc)
                self.log(event='remote_rejected', backend=backend.name, error=str(exc)[:800])
            return
        with self.lock:
            backend.healthy, backend.failures = True, 0
            backend.identity_sha256, backend.identity_error = digest, None
        self.log(event='remote_admitted', backend=backend.name, identity_sha256=digest, runtime=runtime)

    def mark_unhealthy(self, backend, reason):
        with self.lock:
            if not backend.healthy:
                return
            backend.healthy = False
            backend.identity_sha256 = None
            conns = list(backend.conns)
        self.log(event='remote_removed', backend=backend.name, reason=reason, aborted_inflight=len(conns))
        # Abort in-flight requests so they fall back to local immediately
        # instead of waiting for the per-request timeout.
        for conn in conns:
            sock = conn.sock
            if sock is None:
                continue
            # shutdown() wakes a blocked recv on Linux; close() does on Windows.
            for action in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
                try:
                    action()
                except OSError:
                    pass

    def health_loop(self):
        while not self.stop.is_set():
            for backend in self.remotes:
                try:
                    self.check_remote(backend)
                except Exception as exc:   # the monitor must never die
                    self.log(event='health_loop_error', backend=backend.name, error=str(exc)[:300])
            self.stop.wait(self.health_interval)

    # ---------- routing ----------
    def _wait_local_slot(self):
        # self.lock 을 쥔 채로 부른다. 상한이 있으면 자리가 날 때까지 기다린다.
        if self.local_max_inflight is not None:
            while self.local.inflight >= self.local_max_inflight:
                self.local_free.wait(1.0)
                # 기다리는 동안 원격이 돌아오면 원격으로 보낼 수 있게 호출자가 다시 고른다
                if not self.local_routing and any(b.eligible() for b in self.remotes):
                    return False
        self.local.inflight += 1
        return True

    def choose(self):
        with self.lock:
            while True:
                pool = self.all() if self.local_routing else self.remotes
                candidates = [b for b in pool if b.eligible()]
                if candidates:
                    # Least load relative to capacity; local wins ties.
                    best = min(candidates, key=lambda b: (b.inflight / b.capacity, not b.local))
                    best.inflight += 1
                    return best, best.identity_sha256
                if self._wait_local_slot():
                    return self.local, self.local.identity_sha256

    def take_local(self):
        with self.lock:
            while not self._wait_local_slot():
                pass
            return self.local, self.local.identity_sha256

    def release(self, backend):
        with self.lock:
            backend.inflight -= 1
            if backend.local:
                self.local_free.notify_all()

    def _send_raw(self, backend, method, path, headers, body):
        timeout = self.local_timeout if backend.local else self.remote_timeout
        conn = http.client.HTTPConnection(backend.host, backend.port, timeout=self.connect_timeout)
        with self.lock:
            backend.conns.add(conn)
        try:
            conn.connect()
            conn.sock.settimeout(timeout)
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, resp.getheaders(), data
        finally:
            with self.lock:
                backend.conns.discard(conn)
            conn.close()

    def send(self, backend, method, path, headers, body, request_id=None):
        if backend.local:
            return self._send_raw(backend, method, path, headers, body)
        # Remote: wait in slices so a health loss moves the request to local at
        # once on every OS. An abandoned remote answer is dropped, never delivered.
        box, done = {}, threading.Event()

        def worker():
            try:
                box['result'] = self._send_raw(backend, method, path, headers, body)
            except BaseException as exc:
                box['error'] = exc
            finally:
                done.set()
                if box.get('abandoned') and 'result' in box:
                    self.log(event='late_response_discarded', id=request_id, backend=backend.name,
                             response_sha256=sha256(box['result'][2]))

        if not backend.healthy:
            raise ConnectionAbortedError('backend removed before dispatch')
        threading.Thread(target=worker, daemon=True).start()
        while not done.wait(0.2):
            if not backend.healthy:
                box['abandoned'] = True
                raise ConnectionAbortedError('backend removed during request')
        if 'error' in box:
            raise box['error']
        return box['result']

    def forward(self, method, path, headers, body):
        request_id = sha256(os.urandom(16))[:16]
        routed = method == 'POST' and path.split('?')[0] in ROUTED_PATHS
        if not routed:
            # [2026-10-06] 생성이 아닌 요청(모델 목록·상태 확인)은 A100 생성 자리를 차지하지 않는다(doinggyu 검토 5.5).
            # 예전에는 local_max_inflight 자리를 기다려, 실행기의 모델 서버 확인이 생성 요청 뒤에 줄을 섰다.
            status, rheaders, data = self._send_raw(self.local, method, path, headers, body)
            return status, rheaders, data, self.local.name
        backend, identity = self.choose()
        started = time.monotonic()
        fallback_from = error = None
        try:
            try:
                status, rheaders, data = self.send(backend, method, path, headers, body, request_id)
                if not backend.local and status >= 500:
                    raise RuntimeError(f'remote HTTP {status}')
            except Exception as exc:
                if backend.local:
                    raise
                # Transport-level failure on a remote: the same bytes go to local once.
                error = f'{type(exc).__name__}: {exc}'[:300]
                with self.lock:
                    backend.errors += 1
                    backend.fallbacks += 1
                fallback_from = backend.name
                threading.Thread(target=self.mark_unhealthy, args=(backend, 'request_failed'),
                                 daemon=True).start()
                self.release(backend)
                backend, identity = self.take_local()
                status, rheaders, data = self.send(backend, method, path, headers, body)
            with self.lock:
                backend.served += 1
            if routed:
                self.log(event='request', id=request_id, path=path, backend=backend.name,
                         fallback_from=fallback_from, remote_error=error, status=status,
                         seconds=round(time.monotonic() - started, 3),
                         request_sha256=sha256(body or b''), response_sha256=sha256(data),
                         backend_identity_sha256=identity)
            return status, rheaders, data, backend.name
        except Exception as exc:
            with self.lock:
                backend.errors += 1
            self.log(event='request_failed', id=request_id, path=path, backend=backend.name,
                     fallback_from=fallback_from, error=f'{type(exc).__name__}: {exc}'[:300],
                     seconds=round(time.monotonic() - started, 3))
            raise
        finally:
            self.release(backend)

    def status(self):
        with self.lock:
            return {'at_utc': utc(), 'reference_identity_sha256': self.local.identity_sha256,
                    'local_routing': self.local_routing, 'local_max_inflight': self.local_max_inflight,
                    'backends': [b.snapshot() for b in self.all()]}

    def set_drain(self, name, draining):
        for backend in self.remotes:
            if backend.name == name:
                with self.lock:
                    backend.draining = draining
                self.log(event='drain' if draining else 'undrain', backend=name)
                return True
        return False


def make_handler(pool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'gpu-pool/1'

        def log_message(self, *args):
            pass

        def _reply(self, status, headers, data, backend=None):
            self.send_response(status)
            for key, value in headers:
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            if backend:
                self.send_header('X-GPU-Pool-Backend', backend)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, value):
            self._reply(status, [('Content-Type', 'application/json')],
                        json.dumps(value, ensure_ascii=False).encode())

        def _handle(self):
            if self.path.startswith('/pool/'):
                return self._control(self.path)
            if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                return self._json(411, {'error': 'chunked request bodies are not supported'})
            length = self.headers.get('Content-Length')
            body = self.rfile.read(int(length)) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            try:
                status, rheaders, data, name = pool.forward(self.command, self.path, headers, body)
            except Exception as exc:
                return self._json(502, {'error': f'gpu pool: {type(exc).__name__}: {exc}'[:300]})
            self._reply(status, rheaders, data, name)

        def _control(self, path):
            parsed = urllib.parse.urlparse(path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == '/pool/status' and self.command == 'GET':
                return self._json(200, pool.status())
            if parsed.path in ('/pool/drain', '/pool/undrain') and self.command == 'POST':
                name = (query.get('name') or [''])[0]
                ok = pool.set_drain(name, parsed.path == '/pool/drain')
                return self._json(200 if ok else 404, {'ok': ok, 'name': name})
            return self._json(404, {'error': 'unknown pool control path'})

        do_GET = do_POST = do_PUT = do_DELETE = _handle

    return Handler


class PoolHTTPServer(ThreadingHTTPServer):
    # [2026-10-06] 본런은 시뮬 여러 개가 수백 개 요청을 동시에 연다. 기본 대기열 5 칸이면 접속이 밀린다.
    request_queue_size = 1024
    daemon_threads = True


def serve(config):
    pool = Pool(config['local_url'], config.get('remotes', []), config['log_path'],
                **{k: config[k] for k in ('health_interval', 'health_timeout', 'unhealthy_after',
                                          'remote_timeout', 'local_timeout', 'connect_timeout',
                                          'local_routing', 'local_max_inflight')
                   if k in config})
    # Refuse to start unless the local reference server answers.
    deadline = time.monotonic() + float(config.get('startup_wait_seconds', 60))
    while True:
        try:
            pool.load_reference()
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    host = config.get('listen_host', '127.0.0.1')
    server = PoolHTTPServer((host, int(config['listen_port'])), make_handler(pool))
    threading.Thread(target=pool.health_loop, daemon=True).start()
    pool.log(event='proxy_started', pid=os.getpid(), listen=f"{host}:{config['listen_port']}",
             remotes=[r['name'] for r in config.get('remotes', [])])
    return pool, server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = json.load(handle)
    pool, server = serve(config)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        pool.stop.set()
        pool.log(event='proxy_stopped', pid=os.getpid())


if __name__ == '__main__':
    main()
