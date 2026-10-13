"""Generate quirk four-momenta with MadGraph5 and Pythia 8, then run the cut flow.

The heavy fermion is kept stable. Electroweak events are Drell-Yan production of
the charge-1 vector-like lepton ``taup``. Colored events are QCD production of
the charge-2/3 vector-like quark ``tp``. Both include one extra QCD jet and
Pythia 8 MLM matching, which is the generation setup in the paper.

Pythia hadronization is turned off. A stable colored quirk pair does not leave
enough energy for a QCD string to fragment, so Pythia retries that step forever
and never writes an event. The dark string is propagated by this package.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from quirk_search.dynamics import PairState
from quirk_search.kinematics import simulate_from_state
from quirk_search.selection import (
    CUT_FLOW,
    EfficiencyStats,
    _event_row,
    _result_from_rows,
    accumulate_event,
    event_cut_record,
    load_detector_and_cuts,
)

MG5_DIR = Path("/home/waqar/rafia_clone/october_work/MG5_aMC_v3_8_0")
MG5_BIN = MG5_DIR / "bin" / "mg5_aMC"
PROC_DIR = Path("/home/waqar/rafia_clone/october_work/quirk_search/madgraph_proc")

PROCESSES = {
    "ew": {
        "name": "ew_800",
        "mode": "dy",
        "model": "VLLS_NLO",
        "particle": "taup",
        "pdg": 17,
        "mass_gev": 800.0,
        "generate": [
            "generate p p > taup taup~ QED=2 @0",
            "add process p p > taup taup~ j QED=2 QCD=1 @1",
        ],
        "params": [
            "set MTAUP {mass}",
            "set epstaupeL 0",
            "set epstaupmuL 0",
            "set epstauptaL 0",
        ],
    },
    "qcd": {
        "name": "qcd_1800",
        "mode": "qcd",
        "model": "VLQ_UFO",
        "particle": "tp",
        "pdg": 6000006,
        "mass_gev": 1800.0,
        "generate": [
            "generate p p > tp tp~ @0",
            "add process p p > tp tp~ j @1",
        ],
        "params": [
            "set MTP {mass}",
            "set WTP 0",
        ],
    },
}


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open(errors="replace")


def _charge(pdg: int) -> float:
    if abs(pdg) == 17:
        return -1.0 if pdg > 0 else 1.0
    if abs(pdg) == 6000006:
        return 2.0 / 3.0 if pdg > 0 else -2.0 / 3.0
    raise ValueError(f"unsupported pdg {pdg}")


def parse_lhe(path: Path, pdg: int) -> list[dict]:
    events = []
    current: list[tuple] = []
    in_event = False
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("<event"):
                in_event = True
                current = []
                continue
            if stripped.startswith("</event"):
                chosen = [p for p in current if abs(p[0]) == pdg and p[1] == 1]
                if len(chosen) < 2:
                    chosen = [p for p in current if abs(p[0]) == pdg]
                if len(chosen) >= 2:
                    events.append(_pair_record(chosen[0], chosen[1], len(events)))
                in_event = False
                continue
            if not in_event or stripped.startswith("<") or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            try:
                px, py, pz, energy = map(float, parts[0:4])
                this_pdg = int(parts[5] if _looks_like_lhe_pdg(parts) else parts[0])
                status = int(parts[6] if _looks_like_lhe_pdg(parts) else parts[1])
            except ValueError:
                continue
            if _looks_like_lhe_pdg(parts):
                current.append((this_pdg, status, px, py, pz, energy))
            else:
                # HepMC-like P line is handled elsewhere.
                pass
    return events


def _looks_like_lhe_pdg(parts: list[str]) -> bool:
    # LHE particle line: px py pz E mass pdg status ...
    try:
        int(parts[5])
        int(parts[6])
        float(parts[0])
        return True
    except (ValueError, IndexError):
        return False


def parse_hepmc(path: Path, pdg: int) -> list[dict]:
    events = []
    current: list[tuple] = []
    with _open_text(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] in {"E", "HepMC::GenEvent"} and current:
                chosen = _choose_pair(current, pdg)
                if chosen:
                    events.append(_pair_record(chosen[0], chosen[1], len(events)))
                current = []
                continue
            if parts[0] != "P" or len(parts) < 8:
                continue
            try:
                this_pdg = int(float(parts[2]))
                px, py, pz, energy = map(float, parts[3:7])
                status = int(float(parts[8])) if len(parts) > 8 else 1
            except ValueError:
                continue
            current.append((this_pdg, status, px, py, pz, energy))
    if current:
        chosen = _choose_pair(current, pdg)
        if chosen:
            events.append(_pair_record(chosen[0], chosen[1], len(events)))
    return events


def _choose_pair(particles: list[tuple], pdg: int) -> list[tuple] | None:
    matched = [p for p in particles if abs(p[0]) == pdg]
    stable = [p for p in matched if p[1] == 1]
    use = stable if len(stable) >= 2 else matched
    if len(use) < 2:
        return None
    return use[:2]


def _pair_record(first: tuple, second: tuple, index: int) -> dict:
    p1 = np.array(first[2:5], dtype=float)
    p2 = np.array(second[2:5], dtype=float)
    e1 = float(first[5])
    e2 = float(second[5])
    v1 = p1 / e1
    v2 = p2 / e2
    direction = v1 - v2
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    half = 5.0e-4 * direction
    total = p1 + p2
    return {
        "index": index,
        "x1": (half).tolist(),
        "x2": (-half).tolist(),
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "charge1": _charge(first[0]),
        "charge2": _charge(second[0]),
        "pair_pt": float(np.hypot(total[0], total[1])),
        "seed": 170802243 + index * 1_000_003,
    }


def find_event_file(process_dir: Path, pdg: int) -> tuple[Path, list[dict]]:
    events_root = process_dir / "Events"
    candidates = sorted(
        [path for path in events_root.rglob("*") if path.is_file() and ("hepmc" in path.name or path.name.endswith(".lhe") or path.name.endswith(".lhe.gz"))],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if "hepmc" in path.name:
            parsed = parse_hepmc(path, pdg)
        else:
            parsed = parse_lhe(path, pdg)
        if parsed:
            return path, parsed
    raise FileNotFoundError(f"no events containing pdg {pdg} under {events_root}")


def _disable_hadronization(process_dir: Path) -> None:
    """Keep the Pythia shower at parton level.

    MadGraph reads ``Cards/pythia8_card.dat`` when the shower starts. The launch
    menu can edit the run card, but it cannot set this Pythia switch.
    """
    card = process_dir / "Cards" / "pythia8_card.dat"
    if not card.is_file():
        return
    lines = [
        line
        for line in card.read_text().splitlines()
        if not line.strip().lower().startswith("hadronlevel:all")
    ]
    lines.append("HadronLevel:all = off")
    card.write_text("\n".join(lines).rstrip() + "\n")


def _run_mg5(commands: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(
            [str(MG5_BIN)],
            input=commands,
            text=True,
            cwd=str(MG5_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        tail = log_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(f"MadGraph failed, see {log_path}\n{tail}")


def generate_process(kind: str, n_events: int, workers: int, rebuild: bool, log_dir: Path) -> tuple[Path, list[dict]]:
    spec = PROCESSES[kind]
    process_dir = PROC_DIR / spec["name"]
    log_path = log_dir / f"mg5_{spec['name']}.log"
    if rebuild and process_dir.exists():
        shutil.rmtree(process_dir)
    header = ["set automatic_html_opening False", f"set nb_core {workers}"]
    if not (process_dir / "Cards" / "run_card.dat").exists():
        setup = header + [f"import model {spec['model']}", *spec["generate"], f"output {process_dir} -f"]
        _run_mg5("\n".join(setup) + "\n", log_path)
    _disable_hadronization(process_dir)
    commands = header + [
            f"launch {process_dir}",
            "shower=Pythia8",
            "madspin=OFF",
            "analysis=OFF",
            "done",
            f"set nb_core {workers}",
            f"set nevents {n_events}",
            "set ebeam1 6500",
            "set ebeam2 6500",
            "set iseed 170802243",
            "set pdlabel nn23lo1",
            "set ickkw 1",
            "set xqcut 30",
            "set ptj 10",
            "set use_syst False",
    ]
    commands.extend(line.format(mass=spec["mass_gev"]) for line in spec["params"])
    commands.append("done")
    print(f"MadGraph + Pythia: {spec['name']}, {n_events} events, {workers} cores", flush=True)
    _run_mg5("\n".join(commands) + "\n", log_path)
    path, events = find_event_file(process_dir, spec["pdg"])
    events = events[:n_events]
    for index, event in enumerate(events):
        event["index"] = index
        event["seed"] = 170802243 + index * 1_000_003
    print(f"read {len(events)} events from {path}", flush=True)
    return path, events


def _propagate_mg_batch(spec: dict) -> list[dict]:
    detector, cuts = load_detector_and_cuts()
    names = tuple(name for name, _definition in CUT_FLOW)
    rows = []
    for item in spec["events"]:
        rng = np.random.default_rng(int(item["seed"]))
        state = PairState(
            x1=np.array(item["x1"], dtype=float),
            x2=np.array(item["x2"], dtype=float),
            p1=np.array(item["p1"], dtype=float),
            p2=np.array(item["p2"], dtype=float),
            mass_gev=float(spec["mass_gev"]),
            lambda_ev=float(spec["lambda_ev"]),
            charge1=float(item["charge1"]),
            charge2=float(item["charge2"]),
        )
        event = simulate_from_state(
            state,
            detector,
            rng,
            pair_pt_gev=float(item["pair_pt"]),
            mode=str(spec["mode"]),
            pt_trigger_gev=cuts.pt_trigger_gev,
        )
        result = accumulate_event(EfficiencyStats(), event, cuts)
        record = event_cut_record(event, result, cuts)
        rows.append(_event_row(int(item["index"]), event, result, record, names))
    return rows


def propagate_and_analyse(kind: str, events: list[dict], lambda_ev: float, workers: int):
    spec = PROCESSES[kind]
    n_workers = max(1, min(int(workers), len(events)))
    chunks: list[list[dict]] = [[] for _ in range(n_workers)]
    for event in events:
        chunks[event["index"] % n_workers].append(event)
    payloads = [
        {
            "events": chunk,
            "mass_gev": spec["mass_gev"],
            "lambda_ev": float(lambda_ev),
            "mode": spec["mode"],
        }
        for chunk in chunks
        if chunk
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        finished = 0
        for batch in pool.map(_propagate_mg_batch, payloads):
            rows.extend(batch)
            finished += len(batch)
            print(f"{spec['name']}: propagated {finished}/{len(events)}", flush=True)
    return _result_from_rows(rows)
