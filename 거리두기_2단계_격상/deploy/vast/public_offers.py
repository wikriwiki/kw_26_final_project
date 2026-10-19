#!/usr/bin/env python3
"""Read public on-demand offers using the official vast-cli search endpoint.

No API key is read or sent; this cannot rent a server. The public endpoint may
require authentication in the future; use prepare.py search with the CLI then.
Reference: https://github.com/vast-ai/vast-cli/blob/master/vast.py (search__offers)
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

ENDPOINT = "https://console.vast.ai/api/v0/bundles/"
FIELDS = ("id", "gpu_name", "num_gpus", "gpu_ram", "compute_cap", "dph_total", "dph_base",
          "dph_storage", "storage_cost", "inet_down_cost", "inet_up_cost", "inet_down",
          "inet_up", "reliability", "reliability2", "cpu_ram", "cpu_cores_effective",
          "disk_space", "cuda_max_good", "rentable", "rented", "verification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-gpu-gb", type=int, choices=(24, 32, 40, 48, 80), default=32)
    parser.add_argument("--gpu", action="append", help="exact GPU name, e.g. 'RTX 5090'; repeatable")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new snapshot filename")
    # API RAM values use the CLI's GB * 1000 conversion, not the display units.
    query = {"verified": {"eq": True}, "external": {"eq": False},
             "rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1},
             "gpu_ram": {"gte": args.min_gpu_gb * 1000}, "cpu_ram": {"gte": 32000},
             "compute_cap": {"gte": 800}, "cpu_cores_effective": {"gte": 8},
             "disk_space": {"gte": 100}, "reliability": {"gte": 0.98},
             "direct_port_count": {"gte": 1}, "cuda_max_good": {"gte": 12.8},
             "cpu_arch": {"eq": "amd64"}, "gpu_arch": {"eq": "nvidia"},
             "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 100,
             "allocated_storage": 100}
    if args.gpu:
        query["gpu_name"] = {"in": args.gpu}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(query).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            offers = json.load(response)["offers"]
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"Public offer search failed ({type(exc).__name__}); no resources changed.\n")
    if not isinstance(offers, list):
        parser.error("unexpected offer response")
    # Store only relevant public specs and prices; omit machine addresses/location details.
    offers = [{key: offer.get(key) for key in FIELDS} for offer in offers]
    snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": ENDPOINT, "authenticated": False, "rental_type": "on-demand",
                "disk_gb": 100, "query": query, "offers": offers}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    cheapest = {}
    for offer in sorted(offers, key=lambda row: row["dph_total"]):
        cheapest.setdefault(offer["gpu_name"], {key: offer[key] for key in
                            ("id", "gpu_name", "gpu_ram", "dph_total", "inet_down_cost", "inet_up_cost")})
    print(json.dumps({"captured_at_utc": snapshot["captured_at_utc"], "saved": str(args.out),
                      "resources_changed": False, "offer_count": len(offers),
                      "cheapest_per_gpu": list(cheapest.values())}, indent=2))


if __name__ == "__main__":
    main()
