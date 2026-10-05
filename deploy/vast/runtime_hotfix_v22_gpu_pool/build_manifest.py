#!/usr/bin/env python3
"""Write manifest.json for the GPU-pool layer and pin its hash in resume_runtime.py."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ('sitecustomize.py', 'resume_runtime.py', 'onstart.sh'):
        path = HERE / name
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))   # the server runs Linux
    manifest = {
        'schema_version': 1, 'name': 'v22-gpu-pool',
        'source_sha256': 'e6ab8021766613426a615de4f1322b2d315864b857080d4fe047c5e2e4277f7f',
        'previous': {
            'night_progress_manifest_sha256': '19f2a2eaf2587b8ee85da0e5f14278877a86fd4de543346bd31bb325383455eb',
            'night_progress_sitecustomize_sha256': 'b22ba7cc53837a3c9d0194c50a3e4f06f84319f95eeb870add8f588cf976db92',
            'night_progress_controller_sha256': 'c798e9f8d0ff15cad7178639f846ffe3c4c20925833f4fc1039004c1c85701d6'},
        'purpose': 'Route simulator LLM requests through the local inference pool proxy; '
                   'no generation, validation, retry or persistence change',
        'files': {name: digest(HERE / name) for name in ('onstart.sh', 'sitecustomize.py')},
    }
    (HERE / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    manifest_sha = digest(HERE / 'manifest.json')
    controller = HERE / 'resume_runtime.py'
    text = controller.read_text(encoding='utf-8')
    text, count = re.subn(r"MANIFEST_SHA256 = '[0-9a-f_A-Z]*'", f"MANIFEST_SHA256 = '{manifest_sha}'", text)
    if count != 1:
        raise SystemExit('resume_runtime.py must contain exactly one MANIFEST_SHA256 constant')
    controller.write_bytes(text.encode())
    for name in ('manifest.json', 'onstart.sh', 'resume_runtime.py', 'sitecustomize.py'):
        print(digest(HERE / name), name)


if __name__ == '__main__':
    main()
