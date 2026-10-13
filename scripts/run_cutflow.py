#!/usr/bin/env python3
"""Generate electroweak and colored samples and save the sequential cut flow.

This is the bookkeeping for a check of every cut. The default event count is
the 2000 events per sample requested for that check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search import CONFIG_DIR, RESULTS_DIR
from quirk_search.selection import run_sample_parallel
from quirk_search.sensitivity import contour_label, cross_section_pb, expected_events
from quirk_search.yaml_util import load_yaml_file


def _write_events(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=2000, help="Events per sample, before the trigger")
    parser.add_argument("--lambda-ev", type=float, default=4000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16, help="Processes used for one sample")
    args = parser.parse_args()

    config = load_yaml_file(CONFIG_DIR / "benchmarks.yaml")
    base_seed = int(config["seed"] if args.seed is None else args.seed)
    out_dir = RESULTS_DIR / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for offset, chosen in enumerate(config["benchmarks"]):
        print(f"{chosen['name']}: {args.events} events on {args.workers} cores", flush=True)
        sample = run_sample_parallel(
            mass_gev=float(chosen["mass_gev"]),
            lambda_ev=float(args.lambda_ev),
            mode=str(chosen["mode"]),
            n_events=int(args.events),
            seed=base_seed + offset * 100000,
            workers=int(args.workers),
        )
        sigma_pb = cross_section_pb(float(chosen["mass_gev"]), str(chosen["mode"]))
        n_expected = expected_events(
            sigma_pb,
            float(config["luminosity_fb"]),
            sample.cutflow.as_dict()["epsilon_total"],
            float(chosen["charged_fraction"]),
        )
        payload = {
            "benchmark": chosen["name"],
            "mass_gev": float(chosen["mass_gev"]),
            "lambda_ev": float(args.lambda_ev),
            "mode": chosen["mode"],
            "charged_fraction": float(chosen["charged_fraction"]),
            "n_generated": int(args.events),
            "workers": int(args.workers),
            "cross_section_pb": sigma_pb,
            "luminosity_fb": float(config["luminosity_fb"]),
            "expected_events_300fb": n_expected,
            "contour": contour_label(
                n_expected,
                discovery=float(config["discovery_events"]),
                exclusion=float(config["exclusion_events"]),
            ),
            **sample.stats.as_dict(),
            "cutflow": sample.cutflow.as_dict(),
        }
        stem = f"cutflow_{chosen['name']}_lambda{args.lambda_ev:g}_n{args.events}"
        summary_path = out_dir / f"{stem}.json"
        events_path = out_dir / f"{stem}_events.csv"
        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
        _write_events(events_path, sample.events)
        print(json.dumps({"benchmark": chosen["name"], **sample.cutflow.as_dict()}, indent=2))
        print(f"wrote {summary_path}")
        print(f"wrote {events_path}")


if __name__ == "__main__":
    main()
