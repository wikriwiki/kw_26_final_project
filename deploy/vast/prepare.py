#!/usr/bin/env python3
"""Prepare a reviewable Vast rental; never create, modify, or destroy resources.

Only `search --execute` makes a network request, through the official Vast CLI.
All other operations are local. No credential values are read or printed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

# Official PyTorch base; bootstrap installs the pinned EXAONE SGLang fork.
DEFAULT_IMAGE = "pytorch/pytorch@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a"
DEFAULT_QUERY = (
    "verified=true rentable=true rented=false num_gpus=1 "
    "gpu_ram>=32 cpu_ram>=32 cpu_cores_effective>=8 disk_space>=100 "
    "reliability>=0.98 direct_port_count>=1 cuda_vers>=12.8 cpu_arch=amd64 "
    "compute_cap>=800 gpu_arch=nvidia"
)


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def read_preflight() -> dict:
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return {
        "python": sys.version.split()[0],
        "tools": {name: shutil.which(name) is not None for name in ("vastai", "ssh", "scp", "git")},
        "vast_api_key_env_present": bool(os.environ.get("VAST_API_KEY")),
        "vast_api_key_file_present": any(path.is_file() for path in (
            config / "vastai/vast_api_key", Path.home() / ".vast_api_key")),
        "network_called": False,
        "resources_changed": False,
    }


def search_command(args: argparse.Namespace) -> list[str]:
    # --storage must match the eventual rental: it changes the quoted total.
    return ["vastai", "search", "offers", args.query, "--type", "on-demand",
            "--storage", str(args.disk_gb), "--order", "dph_total", "--limit", "20", "--raw"]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting an audit record.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def rental_plan(snapshot: dict, offer_id: int, *, hourly_limit: float,
                hours: float, total_budget: float, transfer_reserve: float,
                image: str = DEFAULT_IMAGE, now: datetime | None = None) -> dict:
    """Validate a fresh quote and produce commands, without executing them."""
    values = (hourly_limit, hours, total_budget, transfer_reserve)
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("all budget inputs, including transfer reserve, must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", image):
        raise ValueError("image must be a Docker reference without shell syntax")
    if ":" not in image or image.endswith(":latest") or "automatic-tag" in image:
        raise ValueError("use a fixed image tag or digest, not latest/automatic-tag")
    captured = datetime.fromisoformat(snapshot["captured_at_utc"])
    if captured.tzinfo is None:
        raise ValueError("offer timestamp must include its UTC timezone")
    now = now or datetime.now(timezone.utc)
    age = (now - captured).total_seconds()
    if not 0 <= age <= 900:
        raise ValueError("offer snapshot is stale; repeat read-only search (15-minute freshness limit)")
    if snapshot.get("rental_type") != "on-demand":
        raise ValueError("only an on-demand quote is supported")
    offers = snapshot["offers"]
    offer = next((o for o in offers if int(o["id"]) == offer_id), None)
    if offer is None:
        raise ValueError("offer ID is absent from the snapshot")
    rate = float(offer["dph_total"])
    disk = float(snapshot["disk_gb"])
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(disk) or disk < 100:
        raise ValueError("quote must contain a positive dph_total and at least 100 GB disk")
    estimate = rate * hours + transfer_reserve
    if rate > hourly_limit:
        raise ValueError("quoted hourly rate exceeds the requested limit")
    if estimate > total_budget:
        raise ValueError("compute/storage estimate plus transfer reserve exceeds budget")
    argv = ["vastai", "create", "instance", str(offer_id), "--image", image,
            "--disk", str(disk), "--ssh", "--direct", "--cancel-unavail",
            "--label", "No_SmokingZone_EXP", "--raw"]
    return {
        "mode": "plan_only_no_resource_changes",
        "offer_id": offer_id,
        "gpu_name": offer.get("gpu_name"),
        "gpu_count": offer.get("num_gpus"),
        "query": snapshot["query"],
        "quoted_hourly_usd_with_requested_storage": rate,
        "planned_rental_hours_including_setup_and_downloads": hours,
        "transfer_reserve_usd": transfer_reserve,
        "estimated_total_usd": round(estimate, 4),
        "total_budget_usd": total_budget,
        "hard_billing_cap": False,
        "image": image,
        "create_argv": argv,
        "create_command": shlex.join(argv),
        "notes": [
            "Refresh the offer immediately before rental; prices and availability can change.",
            "The operator must track elapsed rental time; process timeout does not stop billing.",
            "Export and verify results, then destroy the instance to end storage billing.",
            "Do not map port 8000 publicly. Use an SSH tunnel to localhost.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="check tool/credential presence without reading values")
    search = commands.add_parser("search", help="print a read-only search; --execute fetches offers")
    search.add_argument("--query", default=DEFAULT_QUERY)
    search.add_argument("--disk-gb", type=positive, default=100.0)
    search.add_argument("--execute", action="store_true")
    search.add_argument("--out", type=Path)
    plan = commands.add_parser("plan", help="validate budget and print a rental command, never run it")
    plan.add_argument("--offers", required=True, type=Path)
    plan.add_argument("--offer-id", required=True, type=int)
    plan.add_argument("--max-hourly-usd", required=True, type=positive)
    plan.add_argument("--max-hours", required=True, type=positive)
    plan.add_argument("--total-budget-usd", required=True, type=positive)
    plan.add_argument("--transfer-reserve-usd", required=True, type=positive)
    plan.add_argument("--image", default=DEFAULT_IMAGE)
    plan.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = read_preflight()
        elif args.command == "search":
            command = search_command(args)
            if not args.execute:
                result = {"mode": "dry_run", "argv": command, "command": shlex.join(command)}
            else:
                if not args.out:
                    raise ValueError("--execute requires --out to preserve the quote")
                if args.out.exists():
                    raise ValueError("output exists; choose a new quote filename")
                if not shutil.which("vastai"):
                    raise ValueError("vastai CLI is missing; install it in a local virtual environment")
                run = subprocess.run(command, text=True, encoding="utf-8", capture_output=True,
                                     timeout=90, check=False)
                if run.returncode:
                    # Avoid echoing auth diagnostics that may contain sensitive account values.
                    raise ValueError(f"read-only Vast search failed (exit {run.returncode}); inspect locally")
                payload = json.loads(run.stdout)
                offers = payload.get("offers") if isinstance(payload, dict) else payload
                if not isinstance(offers, list):
                    raise ValueError("unexpected Vast offer response format")
                result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                          "rental_type": "on-demand", "disk_gb": args.disk_gb,
                          "query": args.query, "offers": offers}
                write_json(args.out, result)
                print(json.dumps({"offers_count": len(offers), "saved": str(args.out),
                                  "resources_changed": False}))
                return 0
        else:
            result = rental_plan(json.loads(args.offers.read_text(encoding="utf-8-sig")),
                                 args.offer_id, hourly_limit=args.max_hourly_usd,
                                 hours=args.max_hours, total_budget=args.total_budget_usd,
                                 transfer_reserve=args.transfer_reserve_usd, image=args.image)
            if args.out:
                write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Preparation stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
