#!/usr/bin/env python3
"""Generate MadGraph+Pythia events and save the same cut flow as the report.

Results go to results/madgraph/ and do not overwrite results/benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search.madgraph_events import PROCESSES, generate_process, propagate_and_analyse
from quirk_search.sensitivity import contour_label, cross_section_pb, expected_events
from quirk_search.yaml_util import load_yaml_file
from quirk_search import CONFIG_DIR, RESULTS_DIR


def _write_events(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=2000, help="MadGraph events per sample")
    parser.add_argument("--lambda-ev", type=float, default=4000.0)
    parser.add_argument("--workers", type=int, default=16, help="CPU cores for MadGraph and the plane finder")
    parser.add_argument("--only", choices=["ew", "qcd", "both"], default="both")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate the MadGraph process directories")
    args = parser.parse_args()

    config = load_yaml_file(CONFIG_DIR / "benchmarks.yaml")
    out_dir = RESULTS_DIR / "madgraph"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = ["ew", "qcd"] if args.only == "both" else [args.only]
    for kind in kinds:
        spec = PROCESSES[kind]
        event_path, hard_events = generate_process(
            kind,
            n_events=int(args.events),
            workers=int(args.workers),
            rebuild=bool(args.rebuild),
            log_dir=log_dir,
        )
        if not hard_events:
            raise RuntimeError(f"MadGraph produced no {spec['particle']} pairs for {spec['name']}")
        sample = propagate_and_analyse(
            kind,
            hard_events,
            lambda_ev=float(args.lambda_ev),
            workers=int(args.workers),
        )
        mode = spec["mode"]
        charged = 0.3 if mode == "qcd" else 1.0
        sigma_pb = cross_section_pb(spec["mass_gev"], mode)
        n_expected = expected_events(
            sigma_pb,
            float(config["luminosity_fb"]),
            sample.cutflow.as_dict()["epsilon_total"],
            charged,
        )
        payload = {
            "generator": "madgraph5_amc@nlo 3.8.0 + pythia 8.317",
            "benchmark": spec["name"],
            "model": spec["model"],
            "particle": spec["particle"],
            "mass_gev": spec["mass_gev"],
            "lambda_ev": float(args.lambda_ev),
            "mode": mode,
            "charged_fraction": charged,
            "workers": int(args.workers),
            "madgraph_events_file": str(event_path),
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
        stem = f"cutflow_{spec['name']}_lambda{args.lambda_ev:g}_n{args.events}"
        summary_path = out_dir / f"{stem}.json"
        events_path = out_dir / f"{stem}_events.csv"
        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
        _write_events(events_path, sample.events)
        print(json.dumps({"benchmark": spec["name"], **sample.cutflow.as_dict()}, indent=2))
        print(f"wrote {summary_path}")
        print(f"wrote {events_path}")


if __name__ == "__main__":
    main()
