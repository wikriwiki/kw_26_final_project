#!/usr/bin/env python3
"""Colab 무료(T4) 시험 노트북 — Pro+ 컴퓨팅 단위를 쓰기 전에 G4 와 상관없는 단계를 미리 검사한다 (2026-10-06).

실제 노트북(build_colab_notebook.py)의 칸을 그대로 가져다 쓴다. 바꾸는 것은 T4 에서 막히는 부분뿐이다.
  같은 것  : 패키지 목록(A100 venv) 설치 · EXAONE 정확한 커밋 내려받기·고정 · Secrets 키 복원 · 우리 서버 역방향 터널
  다른 것  : 40GB GPU 검사 없음 / EXAONE 대신 작은 모델(또는 가짜 서버)을 띄운다 — 일부러 A100 과 설정이 다르다.
             서버 중계가 이것을 '설정 다름'으로 거부하고 일을 주지 않는지가 시험 항목이다.
             터널 칸은 끝없이 돌지 않고 20분 뒤 끝난다.
  못 보는 것: EXAONE-33B 가 G4 에서 뜨는지, A100 과 설정 일치 통과, 처리 속도 — Pro+ G4 첫 30분에서 본다.
계정 번호는 6(무료 시험용)을 쓴다. Pro+ 5개는 1~5.

    python deploy/gpu_pool_kw26/build_colab_free_test.py
"""
import hashlib
import json

import build_colab_notebook as prod

OUT = prod.HERE / 'kw26_colab_free_test.ipynb'
TINY = 'Qwen/Qwen2.5-0.5B-Instruct'   # 공개 모델, 약 1GB — 서버 실행 경로 확인용. A100 과 설정이 달라야 정상이다.

INTRO = """
# KW26 — Colab 무료(T4) 시험 노트북

**목적:** Pro+ 컴퓨팅 단위를 쓰기 전에, GPU 종류와 상관없는 단계를 무료 T4 에서 미리 검사한다.

| 칸 | 검사 | 실제 노트북과 |
|---|---|---|
| 1 | GPU·디스크 확인 (40GB 검사 없음) | 다름 |
| 2 | A100 패키지 목록 | 같음 |
| 3 | 설치 + **EXAONE-4.5-33B-AWQ 정확한 커밋 내려받기·고정** | 같음 |
| 4 | 작은 모델로 SGLang 실행(안 되면 가짜 서버) — **일부러 A100 과 설정이 다르다** | 다름 |
| 5 | Secrets 키 복원 → 우리 서버 역방향 터널 → 20분 상태 출력 후 끝 | 같음(끝나는 것만 다름) |

**런타임: T4 GPU.**  계정 6(무료 시험용)이 박혀 있다. Secrets 는 쓰지 않는다 — 키는 런타임 안에서 만들고 공개 쪽만 서버에 등록한다.

**못 보는 것:** EXAONE-33B 가 G4 에서 뜨는지, A100 과 설정 일치 통과, 처리 속도. T4(15GB)에는 33B 가 올라가지 않는다.
"""

CELL_GPU = """
# 1) GPU·디스크 확인 (무료 시험: 40GB 검사 없음)
import os, subprocess, shutil
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
print(out, '| CPU cores', os.cpu_count())
GPU_NAME, GPU_MIB = out.splitlines()[0].split(',')[0].strip(), int(out.splitlines()[0].split(',')[1])
du = shutil.disk_usage('/content'); print('disk free GB', round(du.free / 1e9, 1))
assert du.free > 45e9, '디스크 여유 45GB 미만 — 설치(약 15GB)+모델(약 20GB)이 안 들어간다'
# 터널 포트 두 개(180N1·180N2)를 모두 시험하려고 서버 자리 두 개를 쓴다(G4 는 서버 2개).
PORTS = [8000, 8001]
RESULT = {'gpu': out}
"""

CELL_TEST_SERVER = """
# 4) 시험용 서버 — 작은 모델로 SGLang 을 띄워 본다(T4 에서 안 되면 가짜 서버). 일부러 A100 과 설정이 다르다.
import json, time, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
TINY = '__TINY__'
EXPECTED = __EXPECTED__
SERVER_ENV = dict(os.environ, PATH='/content/venv_sgl/bin:/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
                  CUDA_HOME='/usr/local/cuda', HF_HOME=HF_HOME)
def healthy(port):
    try:
        return urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).status == 200
    except Exception:
        return False
def identity_diff(port):
    info = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/get_server_info', timeout=10))
    return {k: (v, info.get(k, '<absent>')) for k, v in EXPECTED.items() if info.get(k, '<absent>') != v}

class Stub(BaseHTTPRequestHandler):
    # 가짜 서버: 이름부터 EXAONE 이 아니다. 중계가 이 서버에 일을 주면 시험 실패다.
    INFO = {'model_path': 'KW26-FREE-TEST-STUB', 'served_model_name': 'KW26-FREE-TEST-STUB', 'random_seed': 0}
    def _send(self, obj):
        body = json.dumps(obj).encode(); self.send_response(200)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == '/health': return self._send({})
        if self.path == '/get_server_info': return self._send(self.INFO)
        if self.path == '/v1/models': return self._send({'data': [{'id': 'KW26-FREE-TEST-STUB'}]})
        self.send_error(404)
    def do_POST(self):
        self._send({'choices': [{'message': {'content': 'KW26 FREE TEST STUB - this must never reach the simulation'}}]})
    def log_message(self, *a): pass
def start_stub(port):
    srv = ThreadingHTTPServer(('127.0.0.1', port), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(port, 'stub server up, healthy', healthy(port))

ARGS = ['--model-path', TINY, '--host', '127.0.0.1', '--tp-size', '1', '--attention-backend', 'triton',
        '--trust-remote-code', '--reasoning-parser', 'qwen3', '--random-seed', str(SEED)]
port = PORTS[0]
if not healthy(port):
    logpath = f'/content/sglang-{port}.log'; open(logpath, 'w').close()
    proc = subprocess.Popen([VENV, '-m', 'sglang.launch_server', *ARGS, '--port', str(port), '--mem-fraction-static', '0.6'],
                            stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT, start_new_session=True, env=SERVER_ENV)
    t0 = time.time()
    while not healthy(port) and proc.poll() is None and time.time() - t0 < 12 * 60:
        time.sleep(10)
    if healthy(port):
        RESULT['sglang_tiny'] = f'ok after {int(time.time() - t0)}s'
    else:
        tail = open(logpath, errors='replace').read()[-2500:]
        RESULT['sglang_tiny'] = 'FAILED (T4 only? not evidence about G4): ' + tail[-400:]
        print(tail)
        try:
            proc.kill()
        except Exception:
            pass
        start_stub(port)
print('SGLang tiny model:', RESULT.get('sglang_tiny', 'already up'))
start_stub(PORTS[1])
for p in PORTS:
    d = identity_diff(p)
    print(p, '노트북 자체 검사 — A100 과 다른 항목 수', len(d), '(0 이 아니어야 정상)', list(d)[:6])
    assert d, f'{p}: 시험 서버가 A100 과 같은 설정으로 보인다 — 시험을 멈춘다'
RESULT['self_check_mismatch'] = True
"""

TUNNEL_LOOP = """
END_AT = time.time() + 20 * 60      # 무료 시험: 20분 뒤 끝난다
while time.time() < END_AT:
    time.sleep(60)
    ssh_up = bool(subprocess.run(['pgrep', '-f', 'ssh -N -i /root/.ssh/kw26_pool'], capture_output=True).stdout)
    tail = open('/content/tunnel.log', errors='replace').read()[-400:]
    why = '키 거부됨 — Secrets 확인' if 'Permission denied' in tail else ('포트 거부 — 계정 번호 확인' if 'forwarding failed' in tail else 'DOWN(재시도 중)')
    print(time.strftime('%H:%M:%S'), 'health', {p: healthy(p) for p in PORTS}, '| tunnel', 'UP' if ssh_up else why, flush=True)
print('무료 시험 끝 — 결과:', RESULT)
print('tunnel.log 끝:', open('/content/tunnel.log', errors='replace').read()[-600:])
"""


def build():
    freeze = prod.FREEZE.read_text(encoding='utf-8')
    freeze_sha = hashlib.sha256(freeze.encode()).hexdigest()
    install = (prod.CELL_INSTALL.replace('__MODEL__', prod.MODEL).replace('__REVISION__', prod.REVISION)
               .replace('__SEED__', str(prod.SEED)))
    tunnel = (prod.CELL_TUNNEL.replace('__ACCOUNT__', '6').replace('__HOST_KEY__', prod.SERVER_HOST_KEY).replace('__SERVER_PORT__', str(prod.SERVER_PORT))
              .replace('__SERVER_USER__', prod.SERVER_USER).replace('__SERVER_HOST__', prod.SERVER_HOST))
    head, sep, _loop = tunnel.partition('\ndef last_rate(port):')
    assert sep, '실제 노트북 터널 칸의 모양이 바뀌었다 — 시험 노트북을 다시 맞출 것'
    tunnel = head.rstrip('\n') + '\n' + TUNNEL_LOOP
    cells = [prod.md(INTRO), prod.code(CELL_GPU),
             prod.code(prod.CELL_FREEZE.replace('__FREEZE_SHA__', freeze_sha).replace('__FREEZE__', repr(freeze))),
             prod.code(install),
             prod.code(CELL_TEST_SERVER.replace('__TINY__', TINY).replace('__EXPECTED__', repr(prod.EXPECTED))),
             prod.code(tunnel)]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return OUT


if __name__ == '__main__':
    path = build()
    print(path.name, 'sha256', hashlib.sha256(path.read_bytes()).hexdigest())
