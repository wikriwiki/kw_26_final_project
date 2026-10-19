"""Record the served model backing a multi-policy arm without changing it."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def read_proc(pid: int, name: str) -> str:
    return Path(f"/proc/{pid}/{name}").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sim-pid", type=int)
    args = parser.parse_args()
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=8) as r:
        models = json.load(r)
    raw = subprocess.check_output(["ss", "-ltnp"], text=True)
    listening = [line for line in raw.splitlines() if ":8000 " in line]
    if len(listening) != 1:
        raise RuntimeError(f"expected one listener on 8000, found {listening!r}")
    import re
    found = re.search(r"pid=(\d+)", listening[0])
    if not found:
        raise RuntimeError("listener PID unavailable")
    server_pid = int(found.group(1))
    command = read_proc(server_pid, "cmdline")
    served = [item.get("id") for item in models.get("data", [])]
    target = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    if served != [target] or f"--model-path {target}" not in command:
        raise RuntimeError(f"model-server mismatch: served={served!r} command={command!r}")
    evidence = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "listener": listening[0],
        "server_pid": server_pid,
        "server_command": command,
        "served_model_ids": served,
        "launcher_environment": {key: os.environ.get(key) for key in
                                 ("LLM_MODE", "LLM_BASE_URL", "SIM_PROMPT_VARIANT", "SIM_ENVIRONMENT")},
        "note": "Response model_id can reflect the request alias; server command and /v1/models identify the loaded weights.",
    }
    if args.sim_pid:
        env = read_proc(args.sim_pid, "environ").split(" ")
        evidence["sim_pid"] = args.sim_pid
        evidence["sim_command"] = read_proc(args.sim_pid, "cmdline")
        evidence["sim_environment"] = {key: next((x.split("=", 1)[1] for x in env if x.startswith(key + "=")), None)
                                       for key in ("LLM_MODE", "LLM_BASE_URL", "SIM_PROMPT_VARIANT", "SIM_ENVIRONMENT")}
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
