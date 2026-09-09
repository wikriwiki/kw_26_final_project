"""Download public, aggregate reference inputs; never connects to the simulation DB.

Run from any directory with Python 3.10+. Raw files and their SHA256 hashes
are retained so a changed government webpage cannot silently change an input.
"""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/experiments/covid_support_2021"


def fetch(source: dict) -> dict:
    target = DATA / "sources" / source["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = urlencode(source["form"]).encode() if source.get("form") else None
    req = Request(source["url"], data=payload, headers={"User-Agent": "Mozilla/5.0 (research reference collection)"})
    try:
        with urlopen(req, timeout=25) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
        if source["filename"].endswith(".xlsx") and not raw.startswith(b"PK"):
            raise ValueError("Expected XLSX ZIP, received a non-workbook response")
        if not raw:
            raise ValueError("Empty response")
        # Source files are public documents, not executable simulation instructions.
        target.write_bytes(raw)
        return {"id": source["id"], "status": "downloaded", "path": str(target.relative_to(ROOT)).replace("\\", "/"), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "content_type": content_type}
    except Exception as exc:
        return {"id": source["id"], "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Source IDs to refresh; other receipts are preserved")
    args = parser.parse_args()
    catalog = json.loads((DATA / "source_catalog.json").read_text(encoding="utf-8"))
    sources = [s for s in catalog["sources"] if s.get("filename") and (args.only is None or s["id"] in args.only)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, sources))
    now = datetime.now(timezone.utc).isoformat()
    for row in results:
        row["retrieved_at"] = now
    previous_path = DATA / "download_manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))["sources"] if args.only is not None and previous_path.exists() else []
    refreshed = {r["id"] for r in results}
    manifest = {"retrieved_at": now, "sources": [r for r in previous if r["id"] not in refreshed] + results}
    (DATA / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(row["id"], row["status"], row.get("bytes", row.get("error")))


if __name__ == "__main__":
    main()
