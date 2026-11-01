"""웹 콘솔을 이 컴퓨터에서 띄운다 — 시연용 노트북 기준.

    python web/run_local.py --data <데이터 폴더> --prepare   # 처음 한 번 (캐시·지도 저장본 만들기)
    python web/run_local.py --data <데이터 폴더>             # 시연할 때

하는 일 (순서대로)
1. 데이터 폴더(runs·experiments·packs·policies)를 확인한다. 폴더가 흩어져 있으면 --runs 등으로 하나씩 준다.
2. 화면 빌드(web/ui/dist)가 없거나 소스보다 오래됐으면 다시 빌드한다(Node.js 필요).
3. 서버를 띄우고 모든 화면 집계를 미리 계산한다(WEB_WARMUP). 끝날 때까지 기다린다.
4. 다 끝난 뒤에 브라우저를 연다. 첫 클릭부터 바로 뜨게 하려는 순서다.

--prepare 는 3D 지도 저장본(서울 일대, 약 85MB)을 받고, 시민 경험 기록(인터뷰용)을 뽑고,
집계 캐시를 데이터 폴더에 써 둔 뒤 끝난다.
그다음부터는 켜는 데 10초 안팎이다. 자세한 설명은 docs/LOCAL_SETUP.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "web" / "ui"
NEEDED = ("runs", "experiments", "packs", "policies")


def say(message: str) -> None:
    print(message, flush=True)


def find_data(given: str | None) -> Path | None:
    candidates = [given] if given else []
    candidates += [os.environ.get("KW_WEB_DATA"), str(REPO.parent / "kw_web_data"), str(Path.home() / "kw_web_data")]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        # zip 을 풀면 kw_web_data/ 가 한 겹 더 생기는 경우가 있다
        for root in (path, path / "kw_web_data"):
            if any((root / name).is_dir() for name in NEEDED):
                return root
    return None


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """데이터 묶음 하나(--data) 또는 폴더 하나씩(--runs 등). 하나씩 준 값이 이긴다."""
    data = find_data(args.data)
    chosen: dict[str, Path] = {}
    for name in NEEDED:
        given = getattr(args, name)
        if given:
            chosen[name] = Path(given).expanduser().resolve()
        elif data is not None:
            chosen[name] = data / name
    missing = [name for name in NEEDED if name not in chosen or not chosen[name].is_dir()]
    if missing:
        sys.exit(
            "데이터 폴더를 찾지 못했습니다: " + ", ".join(missing) + "\n"
            "  --data <runs·experiments·packs·policies 가 들어 있는 폴더> 로 알려 주거나,\n"
            "  --runs/--experiments/--packs/--policies 로 하나씩 알려 주세요."
        )
    home = data if data is not None else chosen["runs"].parent
    chosen["map_cache"] = Path(args.map_cache).expanduser().resolve() if args.map_cache else home / "map_cache"
    reports = Path(args.reports).expanduser().resolve() if args.reports else None
    if reports is None and (home / "output" / "report").is_dir():
        reports = home / "output"
    if reports is not None and reports.name == "report":
        reports = reports.parent  # 서버는 그 아래 report/ 를 읽는다
    if reports is not None:
        chosen["output"] = reports
    return chosen


def newest(paths: list[Path]) -> float:
    latest = 0.0
    for base in paths:
        if base.is_file():
            latest = max(latest, base.stat().st_mtime)
            continue
        for path in base.rglob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    return latest


def npm_command() -> list[str] | None:
    """윈도의 npm.cmd 는 폴더 이름에 한글이 있으면 작업 위치를 잃는다. node 로 npm 을 직접 부른다."""
    node = shutil.which("node")
    if os.name == "nt" and node:
        cli = Path(node).parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
        if cli.is_file():
            return [node, str(cli)]
    npm = shutil.which("npm")
    return [npm] if npm else None


def ensure_ui(skip: bool) -> None:
    index = UI / "dist" / "index.html"
    sources = [UI / "src", UI / "index.html", UI / "package.json", UI / "vite.config.ts"]
    stale = not index.is_file() or newest(sources) > index.stat().st_mtime
    if not stale:
        return
    if skip:
        say("화면 빌드가 없거나 소스보다 오래됐지만 --skip-build 라 그대로 갑니다.")
        return
    npm = npm_command()
    if not npm:
        sys.exit("화면 빌드(web/ui/dist)가 없거나 오래됐습니다. Node.js 20 이상을 설치한 뒤 다시 실행하세요.")
    say("화면을 빌드합니다 (1~2분)…")
    if not (UI / "node_modules").is_dir():
        subprocess.run([*npm, "ci"], cwd=UI, check=True)
    subprocess.run([*npm, "run", "build"], cwd=UI, check=True)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def health(port: int, path: str = "/api/health") -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
            return json.loads(response.read())
    except Exception:  # noqa: BLE001 — 아직 안 떴다
        return None


def prefetch_map(cache: Path) -> None:
    say(f"3D 지도 저장본을 채웁니다: {cache}")
    result = subprocess.run([sys.executable, str(REPO / "web" / "tools" / "prefetch_map.py"), "--cache", str(cache)], cwd=REPO)
    if result.returncode != 0:
        say("  지도 저장본을 다 채우지 못했습니다. 인터넷이 되는 곳에서 --prepare 를 다시 실행하세요.")


def extract_dossiers(data: Path | None) -> None:
    if data is None or not (data / "zips").is_dir():
        say("시민 경험 기록: 데이터 폴더에 zips/ 가 없어 뽑지 못했습니다. 뽑아 둔 _console/dossier 폴더를 받아 넣으면 됩니다.")
        return
    say("시민 경험 기록(인터뷰용)을 뽑습니다…")
    subprocess.run([sys.executable, str(REPO / "web" / "tools" / "extract_dossier.py"), "--data", str(data)], cwd=REPO)


def has_dossiers(experiments: Path) -> bool:
    return any(experiments.glob("*/_console/dossier/_index.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="웹 콘솔 로컬 실행")
    parser.add_argument("--data", help="runs·experiments·packs·policies 가 들어 있는 폴더 (kw_web_data)")
    for name in NEEDED:
        parser.add_argument(f"--{name}", help=f"{name} 폴더를 따로 줄 때")
    parser.add_argument("--reports", help="보고서 폴더(FINAL_REPORT_*.html 이 있는 곳). 없으면 저장소의 output/sim/report")
    parser.add_argument("--map-cache", help="3D 지도 저장본 폴더 (기본: <데이터 폴더>/map_cache)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    parser.add_argument("--prepare", action="store_true", help="처음 한 번: 지도 저장본·집계 캐시를 만들고 끝낸다")
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 열지 않는다")
    parser.add_argument("--skip-build", action="store_true", help="화면 빌드를 확인하지 않는다")
    parser.add_argument("--demo", action="store_true", help="공개 시연 제한(새 실행·보고서 생성·정책 수정 막기)을 켠다")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 이상이 필요합니다 (지금 {sys.version.split()[0]}).")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        sys.exit(f"패키지가 없습니다. 먼저 실행하세요:\n  {sys.executable} -m pip install -r web/api/requirements.txt")

    paths = resolve_paths(args)
    url = f"http://127.0.0.1:{args.port}"
    if port_in_use(args.port):
        if health(args.port) and not args.prepare:
            say(f"이미 콘솔이 떠 있습니다: {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return
        sys.exit(f"{args.port} 포트를 이미 쓰고 있습니다. 떠 있는 콘솔 창을 닫거나 --port 8770 처럼 다른 번호로 실행하세요.")
    if args.prepare:
        prefetch_map(paths["map_cache"])
        extract_dossiers(find_data(args.data))
    ensure_ui(args.skip_build)

    if "output" not in paths and not any((REPO / "output" / "sim" / "report").glob("FINAL_REPORT_rpt-*.html")):
        say("주의: 보고서 파일이 없어 보고서 화면이 비어 보입니다(git pull 로 받으세요).")
    if not (REPO / "web" / ".env").is_file() and not (REPO / ".env").is_file():
        say("주의: web/.env 가 없어 인터뷰·보고서 해설(K-EXAONE)이 꺼진 채로 뜹니다.")
    if not has_dossiers(paths["experiments"]):
        say("주의: 시민 경험 기록(_console/dossier)이 없어 인터뷰가 간 곳·그때 생각 없이 답합니다. docs/LOCAL_SETUP.md 참고.")
    if not paths["map_cache"].is_dir():
        say("주의: 3D 지도 저장본이 없어 지도를 인터넷에서 받습니다. --prepare 를 한 번 실행하세요.")

    env = {
        **os.environ,
        "SIM_DATA_ROOT": str(paths["runs"]),
        "SIM_EXPERIMENTS_ROOT": str(paths["experiments"]),
        "SIM_VIZ_PACKS_DIR": str(paths["packs"]),
        "SIM_POLICY_DIR": str(paths["policies"]),
        "SIM_MAP_CACHE_DIR": str(paths["map_cache"]),
        # 결과 파일은 시연 중에 바뀌지 않는다. 복사·압축 해제로 파일 시각이 달라져도 크기로 캐시를 맞춘다
        "WEB_CACHE_TRUST_SIZE": "1",
        "WEB_WARMUP": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    if "output" in paths:
        env["SIM_OUTPUT_ROOT"] = str(paths["output"])
    if args.demo:
        env.update({"WEB_PUBLIC_DEMO": "1", "WEB_DEMO_MAX_SAMPLE": "10"})
    command = [
        sys.executable, "-m", "uvicorn", "web.api.app:app",
        "--host", "127.0.0.1", "--port", str(args.port),
        # 요청마다 한 줄씩 찍는 기록은 윈도 콘솔에서 눈에 띄게 느리다
        "--no-access-log", "--log-level", "warning",
    ]
    for name in (*NEEDED, "map_cache", "output"):
        if name in paths:
            say(f"  {name:12s} {paths[name]}")
    say("서버를 띄웁니다…")
    server = subprocess.Popen(command, cwd=REPO, env=env)
    try:
        last = None
        while True:
            if server.poll() is not None:
                sys.exit(f"서버가 시작하지 못했습니다 (종료 코드 {server.returncode}). 위 오류를 확인하세요.")
            status = health(args.port, "/api/warmup")
            warm = status or {}
            if warm.get("state") == "done":
                break
            line = f"미리 계산 중 {warm.get('done', 0)}/{warm.get('total') or '?'}" if status else "서버 시작 중"
            if line != last:
                say(line)
                last = line
            time.sleep(0.5)
        for failure in warm.get("failed") or []:
            say(f"  미리 계산 실패: {failure}")
        if args.prepare:
            say(f"준비 끝 ({warm.get('seconds')}초). 이제 --prepare 없이 실행하면 10초 안팎에 뜹니다.")
            return
        say(f"준비 완료 ({warm.get('seconds')}초). 주소: {url}")
        say("멈추려면 이 창에서 Ctrl+C.")
        if not args.no_browser:
            webbrowser.open(url)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
