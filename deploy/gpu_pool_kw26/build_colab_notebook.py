#!/usr/bin/env python3
"""우리 A100 서버(KW26)에 붙는 Colab 추론 워커 노트북을 만든다.

doinggyu 브랜치(claude/colab-gpu-integration-999c0b)의 deploy/gpu_pool/build_colab_notebook.py 를
우리 서버에 맞게 고친 것이다. 원칙: **시뮬레이션·DB·기록은 우리 서버에만 있고, Colab 은 GPU 만 빌려준다.**
노트북은 DB·Drive·서버 셸 권한을 받지 않는다. 받는 것은 계정 번호와 그 계정 전용 제한 키뿐이다.

우리 서버와 같은 모델 서버를 띄우기 위해 맞추는 것 (A100 의 /get_server_info 에서 읽은 값):
  · 패키지 목록   a100_venv_sgl_freeze.txt (Python 3.12, sglang lkm2835@6757c9f9, torch 2.9.1, transformers 5.8.0)
  · 모델          LGAI-EXAONE/EXAONE-4.5-33B-AWQ, 가중치 커밋 31e6a965 (A100 캐시와 같은 커밋)
  · 실행 인자     --attention-backend triton --trust-remote-code --reasoning-parser qwen3 --random-seed 762156816
                  (A100 은 --revision 없이 떴다 → revision 값이 null 이어야 같은 서버로 인정된다. 그래서 커밋을
                   받은 뒤 캐시의 main 을 그 커밋으로 고정하고 오프라인으로 띄운다.)
A100 의 SGLang 을 다시 띄우면 random_seed 가 바뀐다 — 그때는 SEED 를 새 값으로 바꿔 노트북을 다시 만든다.

    python deploy/gpu_pool_kw26/build_colab_notebook.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = HERE / 'a100_venv_sgl_freeze.txt'
OUT = HERE / 'kw26_colab_sglang_worker.ipynb'

SERVER_HOST, SERVER_PORT, SERVER_USER = '123.37.28.167', 10022, 'outofmemory'
SERVER_HOST_KEY = '[123.37.28.167]:10022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGY7H2+5QX7th9/AMrlHAdkTd8xf/aBH730eKayZTTD4'
MODEL, REVISION, SEED = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ', '31e6a965d0661bbe4a8b895e22a77f8271772ba0', 762156816
# A100 /get_server_info (2026-10-05) 의 생성 관련 값. 프록시도 같은 항목을 다시 검사한 뒤에만 일을 준다.
EXPECTED = {
    'model_path': MODEL, 'served_model_name': MODEL, 'revision': None, 'context_length': None,
    'grammar_backend': 'xgrammar', 'constrained_json_whitespace_pattern': None,
    'constrained_json_disable_any_whitespace': False, 'random_seed': SEED, 'dtype': 'auto',
    'version': '0.0.0.dev11420+g6757c9f90', 'quantization': None, 'kv_cache_dtype': 'auto',
    'trust_remote_code': True, 'chat_template': None, 'tokenizer_path': MODEL, 'tokenizer_mode': 'auto',
    'sampling_defaults': 'model', 'preferred_sampling_params': None, 'json_model_override_args': '{}',
    'reasoning_parser': 'qwen3', 'skip_tokenizer_init': False, 'is_embedding': False,
    'speculative_algorithm': None, 'enable_deterministic_inference': False,
}

INTRO = """
# KW26 — Colab 보조 추론 워커 (GPU 만 빌려준다)

이 노트북은 **언어모델 계산만** 한다. 시뮬레이션·DB·기록은 모두 우리 A100 서버에 있고, 이 노트북은 그것들에 접근할 수 없다.

## 실행 방법
1. **런타임 → 런타임 유형 변경 → G4** (없으면 A100/H100, GPU 메모리 40GB 이상).
2. 왼쪽 🔑 **보안 비밀(Secrets)** 에 두 개를 등록하고 **노트북 액세스**를 켠다.
   - `KW26_POOL_ACCOUNT` : 이 계정에 정해진 번호 **1~6** (계정마다 다른 번호)
   - `KW26_POOL_KEY` : 그 번호의 키 파일 내용 전체 (`kw26_colab_pool_<번호>`, BEGIN 줄부터 END 줄까지)
   - 키는 메신저·GitHub·채팅에 붙여 넣지 않는다. 키 파일에서 Secrets 로 바로 옮긴다.
3. **런타임 → 모두 실행**. 처음에는 설치·모델 내려받기로 15~25분 걸린다.
4. **마지막 셀은 끝나지 않고 계속 돈다. 멈추지 말 것.** 실행 중인 셀이 없으면 Colab 이 약 90분 뒤 세션을 끊는다.
5. Colab 은 연속 **최대 24시간**이다. 끊기면 **모두 실행**을 다시 누른다. 끊겨 있는 동안 시뮬레이션은 멈추지 않는다(느려질 뿐).

## 무엇을 확인하나
- 모델 서버 설정이 우리 A100 과 **한 글자도 다르지 않을 때만** 우리 서버가 일을 준다(노트북도 먼저 스스로 검사한다).
- 마지막 셀이 1분마다 `health {...: True} | tunnel UP | ...` 를 찍으면 정상이다.
"""

CELL_GPU = """
# 1) GPU 확인
import os, subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
print(out, '| CPU cores', os.cpu_count())
GPU_NAME, GPU_MIB = out.splitlines()[0].split(',')[0].strip(), int(out.splitlines()[0].split(',')[1])
assert GPU_MIB >= 40000, f'{GPU_NAME} {GPU_MIB}MiB: G4/H100/A100 런타임으로 바꿔 주세요.'
# 96GB(G4) 이면 서버 2개, 그보다 작으면 1개. 두 번째가 메모리 부족으로 안 뜨면 1개로 계속한다.
N_SERVERS = 2 if GPU_MIB >= 72000 else 1
PER_SERVER_MIB = 40000 if N_SERVERS == 2 else int(GPU_MIB * 0.85)
PORTS = [8000 + i for i in range(N_SERVERS)]
print('model servers:', PORTS, '| per-server budget MiB', PER_SERVER_MIB)
"""

CELL_FREEZE = """
# 2) 우리 A100 `venv_sgl` 의 정확한 패키지 목록 (sha256 __FREEZE_SHA__)
import hashlib, pathlib
FREEZE = __FREEZE__
assert hashlib.sha256(FREEZE.encode()).hexdigest() == '__FREEZE_SHA__'
pathlib.Path('/content/a100_venv_sgl_freeze.txt').write_text(FREEZE)
print('freeze lines', len(FREEZE.splitlines()))
"""

CELL_INSTALL = """
# 3) 설치 + 모델 내려받기 (이미 되어 있으면 건너뜀)
import os, subprocess, sys, pathlib
VENV = '/content/venv_sgl/bin/python'
MODEL, REVISION, SEED = '__MODEL__', '__REVISION__', __SEED__
def sh(cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], sep='\\n')
        r.check_returncode()
    return r.stdout
if not os.path.exists(VENV) or subprocess.run([VENV, '-c', 'import sglang, torch, xgrammar']).returncode:
    sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    UV = [sys.executable, '-m', 'uv']
    sh(UV + ['python', 'install', '3.12'])          # A100 venv 는 Python 3.12
    if not os.path.exists(VENV):
        sh(UV + ['venv', '--python', '3.12', '/content/venv_sgl'])
    # 목록이 정확한 전체 목록이므로 의존성을 다시 계산하지 않고(--no-deps) 그대로 설치한다.
    sh(UV + ['pip', 'install', '--no-deps', '--python', VENV, '-r', '/content/a100_venv_sgl_freeze.txt'])
print(sh([VENV, '-c', 'import importlib.metadata as md, torch; '
          'print("sglang", md.version("sglang"), "| torch", torch.__version__, "| transformers", md.version("transformers"), '
          '"| xgrammar", md.version("xgrammar")); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))']))
HF_HOME = '/content/hf_cache'
os.environ['HF_HOME'] = HF_HOME
sh([VENV, '-c', f"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}', revision='{REVISION}')"],
   env=dict(os.environ, HF_HOME=HF_HOME))
# A100 은 --revision 없이 떴다(서버 정보 revision=null). 같은 커밋을 쓰면서 revision 을 null 로 두려고
# 캐시의 main 을 그 커밋으로 고정하고 서버는 오프라인으로 띄운다.
ref = pathlib.Path(HF_HOME) / 'hub' / ('models--' + MODEL.replace('/', '--')) / 'refs' / 'main'
ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(REVISION)
assert (ref.parent.parent / 'snapshots' / REVISION).is_dir(), '모델 커밋이 캐시에 없다'
print('install + model ok — main ->', REVISION[:12])
"""

CELL_SERVERS = """
# 4) A100 과 같은 인자로 모델 서버 시작 + 설정 일치 확인 (이미 떠 있으면 건너뜀)
import json, time, urllib.request
ARGS = ['--model-path', MODEL, '--host', '127.0.0.1', '--tp-size', '1', '--attention-backend', 'triton',
        '--trust-remote-code', '--reasoning-parser', 'qwen3', '--random-seed', str(SEED)]
EXPECTED = __EXPECTED__
# 가중치 변환 커널을 즉석 컴파일하므로 venv 의 ninja 와 CUDA nvcc 가 PATH 에 있어야 한다(doinggyu 운영 기록).
SERVER_ENV = dict(os.environ, PATH='/content/venv_sgl/bin:/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
                  CUDA_HOME='/usr/local/cuda', HF_HOME=HF_HOME, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
def healthy(port):
    try:
        return urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).status == 200
    except Exception:
        return False
def fraction_for(target_mib):
    # --mem-fraction-static 은 GPU 를 혼자 쓴다고 가정한다. 다른 서버가 쓰는 만큼을 빼고 계산한다.
    free, total = [int(x) for x in sh(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                                       '--format=csv,noheader,nounits']).split(',')]
    assert free - target_mib > 3000, f'GPU 여유 부족: free {free} MiB'
    return round(1 - (free - target_mib) / total, 3)
def check_identity(port):
    info = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/get_server_info', timeout=10))
    diff = {k: (v, info.get(k, '<absent>')) for k, v in EXPECTED.items() if info.get(k, '<absent>') != v}
    assert not diff, f'{port} A100 과 설정이 다르다: {diff}'
    return info.get('max_total_num_tokens')
def start_server(port):
    if healthy(port):
        print(port, 'already healthy | KV tokens', check_identity(port)); return
    logpath = f'/content/sglang-{port}.log'
    open(logpath, 'w').close()
    fraction = fraction_for(PER_SERVER_MIB)
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
start_server(PORTS[0])
if len(PORTS) > 1:
    try:
        start_server(PORTS[1])
    except Exception as exc:
        print('두 번째 서버를 못 띄웠다 — 서버 1개로 계속한다:', str(exc)[:300])
        PORTS = PORTS[:1]
print(sh(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader']).strip())
"""

CELL_TUNNEL = """
# 5) 우리 서버로 역방향 터널 + 상태 루프.  ★ 이 셀은 끝나지 않는다 — 멈추지 말 것 ★
import pathlib, re
from google.colab import userdata
ACCOUNT = int(str(userdata.get('KW26_POOL_ACCOUNT')).strip())
assert 1 <= ACCOUNT <= 6, 'KW26_POOL_ACCOUNT 는 1~6'
raw = userdata.get('KW26_POOL_KEY')
# Secrets 는 여러 줄 값을 한 줄로 합쳐 저장한다 → 개인키의 줄바꿈을 복원한다. 키 본문은 출력하지 않는다.
m = re.search(r'-----BEGIN OPENSSH PRIVATE KEY-----(.*?)-----END OPENSSH PRIVATE KEY-----', raw, re.S)
assert m, 'KW26_POOL_KEY 에 키 파일 전체(BEGIN~END)를 넣어 주세요.'
body = re.sub(r'\\s+', '', m.group(1))
ssh_dir = pathlib.Path('/root/.ssh'); ssh_dir.mkdir(mode=0o700, exist_ok=True)
(ssh_dir / 'kw26_pool').write_text('-----BEGIN OPENSSH PRIVATE KEY-----\\n'
    + '\\n'.join(body[i:i + 70] for i in range(0, len(body), 70)) + '\\n-----END OPENSSH PRIVATE KEY-----\\n')
os.chmod(ssh_dir / 'kw26_pool', 0o600)
print('key fingerprint:', sh(['ssh-keygen', '-lf', '/root/.ssh/kw26_pool']).strip())
(ssh_dir / 'known_hosts').write_text('__HOST_KEY__\\n')
# 이 키는 우리 서버에서 셸이 막혀 있고, 이 계정 번호의 포트 두 개(180N1, 180N2)만 열 수 있다.
forwards = ' '.join(f'-R 127.0.0.1:180{ACCOUNT}{i + 1}:127.0.0.1:{p}' for i, p in enumerate(PORTS))
subprocess.run(['pkill', '-f', '/content/tunnel.sh']); subprocess.run(['pkill', '-f', 'ssh -N -i /root/.ssh/kw26_pool'])
pathlib.Path('/content/tunnel.sh').write_text(f'''while true; do
  ssh -N -i /root/.ssh/kw26_pool -p __SERVER_PORT__ -o StrictHostKeyChecking=yes -o IdentitiesOnly=yes \\\\
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 {forwards} __SERVER_USER__@__SERVER_HOST__
  echo "$(date -u +%FT%TZ) tunnel exited $?"; sleep 5
done
''')
open('/content/tunnel.log', 'w').close()
subprocess.Popen(['bash', '/content/tunnel.sh'], stdout=open('/content/tunnel.log', 'ab'),
                 stderr=subprocess.STDOUT, start_new_session=True)
print('account', ACCOUNT, '| tunnel started:', forwards, flush=True)

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
    ssh_up = bool(subprocess.run(['pgrep', '-f', 'ssh -N -i /root/.ssh/kw26_pool'], capture_output=True).stdout)
    tail = open('/content/tunnel.log', errors='replace').read()[-400:]
    why = '키 거부됨 — Secrets 확인' if 'Permission denied' in tail else ('포트 거부 — 계정 번호 확인' if 'forwarding failed' in tail else 'DOWN(재시도 중)')
    print(time.strftime('%H:%M:%S'), 'health', {p: healthy(p) for p in PORTS},
          '| tunnel', 'UP' if ssh_up else why, '|', [last_rate(p) for p in PORTS],
          '| gpu', sh(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], check=False).strip(), flush=True)
"""


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


def build():
    freeze = FREEZE.read_text(encoding='utf-8')
    freeze_sha = hashlib.sha256(freeze.encode()).hexdigest()
    install = (CELL_INSTALL.replace('__MODEL__', MODEL).replace('__REVISION__', REVISION)
               .replace('__SEED__', str(SEED)))
    tunnel = (CELL_TUNNEL.replace('__HOST_KEY__', SERVER_HOST_KEY).replace('__SERVER_PORT__', str(SERVER_PORT))
              .replace('__SERVER_USER__', SERVER_USER).replace('__SERVER_HOST__', SERVER_HOST))
    cells = [md(INTRO), code(CELL_GPU),
             code(CELL_FREEZE.replace('__FREEZE_SHA__', freeze_sha).replace('__FREEZE__', repr(freeze))),
             code(install), code(CELL_SERVERS.replace('__EXPECTED__', repr(EXPECTED))), code(tunnel)]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return OUT, freeze_sha


if __name__ == '__main__':
    path, digest = build()
    print(path.name, 'freeze_sha256', digest, 'notebook_sha256', hashlib.sha256(path.read_bytes()).hexdigest())
