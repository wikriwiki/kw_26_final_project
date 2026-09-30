#!/usr/bin/env python3
"""Build the Colab worker notebook that serves the *same* SGLang model to Vast.

The notebook never receives DB, Drive or root credentials. It rebuilds the
exact Vast SGLang environment from the pinned freeze, starts identical model
servers on one large GPU, checks their generation identity, opens reverse
tunnels with a key that Vast restricts to those loopback ports, and then keeps
one cell running (Colab disconnects a notebook with no executing cell).

Everything here was exercised on a Colab G4 (RTX PRO 6000 Blackwell 96GB) on
2026-09-30; the comments in the cells record what was needed to make it work.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = HERE / 'vast_venv_sgl_freeze.txt'
OUT = HERE / 'colab_sglang_worker.ipynb'
VAST_HOST_KEY = '[95.3.33.46]:45025 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEgR1PERp5jaFe4VbxWOT5rAgXKU9uEA1fEI3MAS49pK'

INTRO = r'''
# No_SmokingZone_EXP — Colab 보조 추론 워커

이 노트북은 **추론만** 한다. DB·Drive·Vast root 권한을 받지 않는다. 결과 저장·검증·재시도 횟수는 모두 Vast가 맡는다.

## 실행 방법
1. 런타임 유형을 **G4 (또는 H100/A100, VRAM 40GB 이상)** 로 연결한다.
2. 왼쪽 🔑 **Secrets** 에 아래 3개를 등록하고 **노트북 액세스**를 켠다 (노트북을 새로 올리면 스위치를 다시 켜야 한다).
   - `VAST_GPU_POOL_KEY` : PC의 `~/.ssh/no_smoking_gpu_pool` 파일 내용 전체
   - `VAST_SSH_HOST` : `95.3.33.46`
   - `VAST_SSH_PORT` : `45025`
3. **런타임 → 모두 실행**. 처음에는 설치·모델 다운로드로 10~20분 걸린다.
4. **마지막 셀은 끝나지 않고 계속 돈다. 멈추지 말 것.** 실행 중인 셀이 없으면 Colab이 약 90분 뒤 유휴로 세션을 끊는다.

세션이 끊겨도 실험은 멈추지 않는다 (Vast가 남은 요청을 이어받는다). 다시 가속하려면 **모두 실행**을 다시 누르면 된다. 이미 끝난 단계는 자동으로 건너뛴다.
'''

CELL_GPU = r'''
# 1) GPU 확인
import os, subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
print(out, '| CPU cores', os.cpu_count())
GPU_NAME, GPU_MIB = out.splitlines()[0].split(',')[0].strip(), int(out.splitlines()[0].split(',')[1])
assert GPU_MIB >= 40000, f'{GPU_NAME} {GPU_MIB}MiB: G4/H100/A100 런타임으로 바꿔 주세요.'
# 서버 1개에 약 30GB. G4에서 1개→2개는 +16%였고 3개째는 메모리 여유가 부족했다 → 최대 2개.
N_SERVERS = 2 if GPU_MIB >= 72000 else 1
PORTS = [8000 + i for i in range(N_SERVERS)]
print('model servers:', PORTS)
'''

CELL_FREEZE = r'''
# 2) Vast `venv_sgl`의 정확한 패키지 목록 (sha256 __FREEZE_SHA__)
import hashlib, pathlib
FREEZE = __FREEZE__
assert hashlib.sha256(FREEZE.encode()).hexdigest() == '__FREEZE_SHA__'
pathlib.Path('/content/vast_venv_sgl_freeze.txt').write_text(FREEZE)
print('freeze lines', len(FREEZE.splitlines()))
'''

CELL_INSTALL = r'''
# 3) 설치 + 모델 다운로드 (이미 되어 있으면 건너뜀)
import os, subprocess, sys
VENV = '/content/venv_sgl/bin/python'
MODEL, REVISION = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ', '31e6a965d0661bbe4a8b895e22a77f8271772ba0'
def sh(cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], sep='\n')
        r.check_returncode()
    return r.stdout
if not os.path.exists(VENV) or subprocess.run([VENV, '-c', 'import sglang, torch, outlines']).returncode:
    sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    UV = [sys.executable, '-m', 'uv']
    sh(UV + ['python', 'install', '3.11'])          # Vast venv는 Python 3.11
    if not os.path.exists(VENV):
        sh(UV + ['venv', '--python', '3.11', '/content/venv_sgl'])
    # Vast는 sglang 설치 뒤 transformers를 5.8.0으로 올렸다. freeze가 정확한 전체 목록이므로
    # 의존성을 다시 계산하지 않고(--no-deps) 그대로 설치해야 같은 환경이 된다.
    sh(UV + ['pip', 'install', '--no-deps', '--python', VENV, '-r', '/content/vast_venv_sgl_freeze.txt'])
print(sh([VENV, '-c', 'import importlib.metadata as md, torch; '
          'print("sglang", md.version("sglang"), "| torch", torch.__version__, "| transformers", md.version("transformers"), '
          '"| outlines", md.version("outlines"), "| sglang-kernel", md.version("sglang-kernel")); '
          'print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))']))
sh([VENV, '-c', f"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}', revision='{REVISION}')"])
print('install + model ok')
'''

CELL_SERVERS = r'''
# 4) Vast에 기록된 것과 같은 인자로 모델 서버 시작 + 생성 동일성 확인 (이미 떠 있으면 건너뜀)
import json, time, urllib.request
ARGS = ['--model-path', MODEL, '--host', '127.0.0.1', '--served-model-name', MODEL, '--dtype', 'auto',
        '--context-length', '16384', '--max-running-requests', '16', '--tp-size', '1',
        '--attention-backend', 'triton', '--revision', REVISION, '--random-seed', '42',
        '--trust-remote-code', '--grammar-backend', 'outlines',
        '--constrained-json-whitespace-pattern', '[\\n\\t ]*']
# Vast의 /get_server_info 값. Vast 프록시도 같은 항목을 다시 검사한 뒤에만 작업을 준다.
EXPECTED = {'model_path': MODEL, 'served_model_name': MODEL, 'revision': REVISION, 'context_length': 16384,
            'grammar_backend': 'outlines', 'constrained_json_whitespace_pattern': '[\\n\\t ]*',
            'random_seed': 42, 'dtype': 'auto', 'version': '0.0.0.dev11420+g6757c9f90',
            'quantization': None, 'kv_cache_dtype': 'auto', 'trust_remote_code': True}
# 가중치 변환 커널을 즉석 컴파일하므로 venv의 ninja와 CUDA nvcc(Vast와 같은 12.8)가 PATH에 있어야 한다.
SERVER_ENV = dict(os.environ, PATH='/content/venv_sgl/bin:/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
                  CUDA_HOME='/usr/local/cuda')
print(sh(['nvcc', '--version'], env=SERVER_ENV).strip().splitlines()[-2])

def healthy(port):
    try:
        return urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).status == 200
    except Exception:
        return False

def fraction_for(target_mib=30500):
    # --mem-fraction-static은 GPU를 혼자 쓴다고 가정한다. 다른 서버가 이미 쓰는 만큼을 빼고
    # 이 서버의 예산이 target이 되도록 남은 메모리 기준으로 계산한다.
    free, total = [int(x) for x in sh(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                                       '--format=csv,noheader,nounits']).split(',')]
    assert free - target_mib > 3000, f'GPU 여유 부족: free {free} MiB'
    return round(1 - (free - target_mib) / total, 3)

def check_identity(port):
    info = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/get_server_info', timeout=10))
    diff = {k: (v, info.get(k)) for k, v in EXPECTED.items() if info.get(k) != v}
    assert not diff, f'{port} identity mismatch: {diff}'
    return info.get('max_total_num_tokens')

def start_server(port):
    if healthy(port):
        print(port, 'already healthy | KV tokens', check_identity(port)); return
    logpath = f'/content/sglang-{port}.log'
    open(logpath, 'w').close()
    fraction = fraction_for()
    proc = subprocess.Popen([VENV, '-m', 'sglang.launch_server', *ARGS, '--port', str(port),
                             '--mem-fraction-static', str(fraction)],
                            stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT, start_new_session=True, env=SERVER_ENV)
    t0 = time.time()
    while not healthy(port):
        if proc.poll() is not None:
            print(open(logpath, errors='replace').read()[-5000:])
            raise RuntimeError(f'SGLang {port} exited (code {proc.returncode})')
        assert time.time() - t0 < 40 * 60, f'{port} not healthy in 40 min'
        time.sleep(15)
    print(port, f'healthy after {int(time.time() - t0)}s | mem_fraction_static {fraction} | KV tokens', check_identity(port), flush=True)

for port in PORTS:
    start_server(port)
print(sh(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader']).strip())
'''

CELL_BENCH = r'''
# 5) (선택) 소량 성능 시험 — 본 실험 결과에 포함되지 않는다. 필요할 때만 RUN_BENCH = True
RUN_BENCH = False
if RUN_BENCH:
    import concurrent.futures
    PROMPT = ('서울 송파구에 사는 40대 직장인의 과거 일주일 기록이다. 월요일 출근 후 점심은 회사 근처 식당, 퇴근 후 헬스장. '
              '화요일은 야근 후 편의점. 수요일은 동료와 저녁 회식 후 스크린골프. 목요일은 재택, 동네 카페. ') * 128
    def one(i):
        body = json.dumps({'model': MODEL, 'temperature': 0.0, 'seed': 1000 + i, 'max_tokens': 1100,
            'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': 'Answer only with one JSON object.'},
                         {'role': 'user', 'content': PROMPT + f'\n위 기록으로 오늘 하루 계획 20개 항목을 JSON으로 작성하라. #{i}'}]}).encode()
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{PORTS[i % len(PORTS)]}/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=1200))
        return time.time() - t, r['usage']['prompt_tokens'], r['usage']['completion_tokens']
    for n in (len(PORTS), 6):
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            rows = list(ex.map(one, range(n)))
        wall, out_tok = time.time() - t0, sum(r[2] for r in rows)
        print(f'concurrent={n}: in≈{rows[0][1]} tok | out {out_tok} tok | total {out_tok / wall:.1f} tok/s | '
              f'per-request {sum(r[2] / r[0] for r in rows) / n:.1f} tok/s')
'''

CELL_TUNNEL = r'''
# 6) Vast로 역방향 터널 + 상태 루프.  ★ 이 셀은 끝나지 않는다 — 멈추지 말 것 ★
#    (실행 중인 셀이 없으면 Colab이 약 90분 뒤 유휴로 세션을 끊는다)
import pathlib, re
from google.colab import userdata
raw = userdata.get('VAST_GPU_POOL_KEY')
host, ssh_port = userdata.get('VAST_SSH_HOST').strip(), userdata.get('VAST_SSH_PORT').strip()
# Secrets는 여러 줄 값을 한 줄로 합쳐 저장한다 → 개인키의 줄바꿈을 복원한다. 키 본문은 출력하지 않는다.
m = re.search(r'-----BEGIN OPENSSH PRIVATE KEY-----(.*?)-----END OPENSSH PRIVATE KEY-----', raw, re.S)
assert m, 'VAST_GPU_POOL_KEY 에 키 파일 전체(BEGIN~END)를 넣어 주세요.'
body = re.sub(r'\s+', '', m.group(1))
ssh_dir = pathlib.Path('/root/.ssh'); ssh_dir.mkdir(mode=0o700, exist_ok=True)
(ssh_dir / 'gpu_pool').write_text('-----BEGIN OPENSSH PRIVATE KEY-----\n'
    + '\n'.join(body[i:i + 70] for i in range(0, len(body), 70)) + '\n-----END OPENSSH PRIVATE KEY-----\n')
os.chmod(ssh_dir / 'gpu_pool', 0o600)
print('key fingerprint:', sh(['ssh-keygen', '-lf', '/root/.ssh/gpu_pool']).strip())
pinned = '__VAST_HOST_KEY__'
if pinned.startswith(f'[{host}]:{ssh_port} '):
    (ssh_dir / 'known_hosts').write_text(pinned + '\n'); host_check = 'yes'
else:
    host_check = 'accept-new'     # Vast 주소가 바뀐 경우: 첫 접속의 호스트 키를 기록
# 이 키는 Vast에서 셸이 막혀 있고 127.0.0.1:18001~18004 리슨만 허용된다.
forwards = ' '.join(f'-R 127.0.0.1:{18001 + i}:127.0.0.1:{p}' for i, p in enumerate(PORTS))
subprocess.run(['pkill', '-f', '/content/tunnel.sh']); subprocess.run(['pkill', '-f', 'ssh -N -i /root/.ssh/gpu_pool'])
pathlib.Path('/content/tunnel.sh').write_text(f"""while true; do
  ssh -N -i /root/.ssh/gpu_pool -p {ssh_port} -o StrictHostKeyChecking={host_check} -o ExitOnForwardFailure=yes \\
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 {forwards} root@{host}
  echo "$(date -u +%FT%TZ) tunnel exited $?"; sleep 5
done
""")
open('/content/tunnel.log', 'w').close()
subprocess.Popen(['bash', '/content/tunnel.sh'], stdout=open('/content/tunnel.log', 'ab'),
                 stderr=subprocess.STDOUT, start_new_session=True)
print('tunnel started:', forwards, flush=True)

def last_rate(port):
    lines = [l for l in open(f'/content/sglang-{port}.log', errors='replace').read()[-8000:].splitlines() if 'gen throughput' in l]
    if not lines:
        return '-'
    return (lines[-1].split('#running-req:')[1].split(',')[0].strip() + 'req/'
            + lines[-1].split('gen throughput (token/s):')[1].split(',')[0].strip() + 'tok/s')

while True:
    time.sleep(60)
    for port in PORTS:                       # 서버가 죽었으면 다시 올린다
        if not healthy(port):
            print(time.strftime('%H:%M:%S'), port, 'DOWN → restart', flush=True)
            try:
                start_server(port)
            except Exception as exc:
                print('restart failed:', str(exc)[:200], flush=True)
    ssh_up = bool(subprocess.run(['pgrep', '-f', 'ssh -N -i /root/.ssh/gpu_pool'], capture_output=True).stdout)
    denied = 'Permission denied' in open('/content/tunnel.log', errors='replace').read()[-400:]
    print(time.strftime('%H:%M:%S'), 'health', {p: healthy(p) for p in PORTS},
          '| tunnel', 'UP' if ssh_up else ('키 거부됨 — Secrets 확인' if denied else 'DOWN(재시도 중)'),
          '|', [last_rate(p) for p in PORTS],
          '| gpu', sh(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], check=False).strip(), flush=True)
'''


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


def build():
    freeze = FREEZE.read_text(encoding='utf-8')
    freeze_sha = hashlib.sha256(freeze.encode()).hexdigest()
    cells = [md(INTRO), code(CELL_GPU),
             code(CELL_FREEZE.replace('__FREEZE_SHA__', freeze_sha).replace('__FREEZE__', repr(freeze))),
             code(CELL_INSTALL), code(CELL_SERVERS), code(CELL_BENCH),
             code(CELL_TUNNEL.replace('__VAST_HOST_KEY__', VAST_HOST_KEY))]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return OUT, freeze_sha


if __name__ == '__main__':
    path, digest = build()
    print(path.name, 'freeze_sha256', digest, 'notebook_sha256', hashlib.sha256(path.read_bytes()).hexdigest())
