"""Signal region and efficiency breakdown of Sec. IV.

epsilon = epsilon_trigger * epsilon_fiducial * epsilon_reconstruction.

The reconstruction numerator is the Sec. IV signal region: a plane from the
finder, Delta s < 0.01 cm, Delta w < 1 cm, and two hits on every layer except
possibly one layer that still has at least one hit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from quirk_search.detector import Detector, detector_from_config
from quirk_search.kinematics import SignalEvent, simulate_signal_event
from quirk_search.plane_finder import PaperCuts, PlaneResult, find_plane
from quirk_search.yaml_util import load_yaml_file
from quirk_search import CONFIG_DIR


def cuts_from_config(path=None) -> PaperCuts:
    path = CONFIG_DIR / "selection.yaml" if path is None else path
    raw = load_yaml_file(path)
    return PaperCuts(
        delta_phi=float(raw["delta_phi"]),
        delta_z_mm=float(raw["delta_z_mm"]),
        delta_s_seed_mm=float(raw["delta_s_seed_mm"]),
        delta_w_seed_mm=float(raw["delta_w_seed_mm"]),
        delta_s_iter_mm=float(raw["delta_s_iter_mm"]),
        delta_w_iter_mm=float(raw["delta_w_iter_mm"]),
        grow_factor=float(raw["grow_factor"]),
        delta_s_final_mm=float(raw["delta_s_final_mm"]),
        delta_w_final_mm=float(raw["delta_w_final_mm"]),
        seed_pair_phi=float(raw.get("seed_pair_phi", 0.5)),
        seed_pair_z_mm=float(raw.get("seed_pair_z_mm", 80.0)),
        n_layers=int(raw.get("n_layers", 8)),
        pt_trigger_gev=float(raw["pt_trigger_gev"]),
    )


def hit_pattern_ok(layer_counts: np.ndarray) -> bool:
    """All but one layer has two hits; the remaining layer has at least one."""
    counts = np.asarray(layer_counts)
    if counts.size != 8:
        return False
    if np.any(counts < 1):
        return False
    return int(np.sum(counts < 2)) <= 1


# Sequential cuts. Each later cut is counted only for events that passed every earlier one.
# The three efficiencies are ratios of these counts, as defined in the paper's Eq. (12).
CUT_FLOW = (
    (
        "generated",
        "Every simulated event, before any selection.",
    ),
    (
        "trigger",
        "Pair transverse momentum above 200 GeV. "
        "epsilon_trigger = N(trigger) / N(generated).",
    ),
    (
        "fiducial",
        "Both quirks intersect all eight barrel layers, and every hit time is under 25 ns. "
        "epsilon_fiducial = N(fiducial) / N(trigger).",
    ),
    (
        "seed_pair_layer8",
        "Layer 8 has a hit pair with Delta phi < 0.1 and Delta z < 2 cm.",
    ),
    (
        "seed_pair_layer7",
        "Layer 7 has a hit pair with the same Delta phi and Delta z window.",
    ),
    (
        "seed_prefilter",
        "The layer-8 pair and the layer-7 pair agree within 0.5 rad and 80 mm. "
        "This drops combinations that cannot satisfy the 1 cm width cut.",
    ),
    (
        "seed_delta_s",
        "The four seed hits have plane thickness Delta s < 0.05 cm.",
    ),
    (
        "seed_delta_w",
        "The four seed hits have oscillation width Delta w < 1 cm.",
    ),
    (
        "seed_forward",
        "Every seed hit lies in the forward half of the fitted direction, n3 · x > 0.",
    ),
    (
        "inward_walk",
        "Hits are added from layer 6 down to layer 1 only when both Delta s and Delta w "
        "stay within a factor of three, and the finished plane has Delta w < 1 cm.",
    ),
    (
        "final_delta_s",
        "The finished plane has thickness Delta s < 0.01 cm.",
    ),
    (
        "hit_pattern",
        "Seven layers contain two hits and the remaining layer contains at least one. "
        "epsilon_reco = N(hit_pattern) / N(fiducial). "
        "epsilon_total = epsilon_trigger * epsilon_fiducial * epsilon_reco.",
    ),
)


def passes_signal_region(result: PlaneResult, cuts: PaperCuts | None = None) -> bool:
    cuts = PaperCuts() if cuts is None else cuts
    if not result.found:
        return False
    if result.delta_w_mm >= cuts.delta_w_final_mm:
        return False
    if result.delta_s_mm >= cuts.delta_s_final_mm:
        return False
    return hit_pattern_ok(result.layer_counts)


@dataclass
class EfficiencyStats:
    n_generated: int = 0
    n_triggered: int = 0
    n_fiducial: int = 0
    n_reconstructed: int = 0

    def add(self, triggered: bool, fiducial: bool, reconstructed: bool) -> None:
        self.n_generated += 1
        if not triggered:
            return
        self.n_triggered += 1
        if not fiducial:
            return
        self.n_fiducial += 1
        if reconstructed:
            self.n_reconstructed += 1

    @property
    def epsilon_trigger(self) -> float:
        return self.n_triggered / self.n_generated if self.n_generated else 0.0

    @property
    def epsilon_fiducial(self) -> float:
        return self.n_fiducial / self.n_triggered if self.n_triggered else 0.0

    @property
    def epsilon_reco(self) -> float:
        return self.n_reconstructed / self.n_fiducial if self.n_fiducial else 0.0

    @property
    def epsilon_total(self) -> float:
        return self.epsilon_trigger * self.epsilon_fiducial * self.epsilon_reco

    def as_dict(self) -> dict:
        return {
            "n_generated": self.n_generated,
            "n_triggered": self.n_triggered,
            "n_fiducial": self.n_fiducial,
            "n_reconstructed": self.n_reconstructed,
            "epsilon_trigger": self.epsilon_trigger,
            "epsilon_fiducial": self.epsilon_fiducial,
            "epsilon_reco": self.epsilon_reco,
            "epsilon_total": self.epsilon_total,
        }


def event_cut_record(event: SignalEvent, result: PlaneResult, cuts: PaperCuts) -> dict[str, bool]:
    """Boolean outcome of every cut for one event, before the sequential requirement."""
    flags = result.cut_flags
    final_ds = bool(result.found) and result.delta_s_mm < cuts.delta_s_final_mm
    pattern = bool(result.found) and hit_pattern_ok(result.layer_counts)
    return {
        "generated": True,
        "trigger": bool(event.triggered),
        "fiducial": bool(event.measured.fiducial(cuts.n_layers)),
        "seed_pair_layer8": bool(flags.get("seed_pair_layer8", False)),
        "seed_pair_layer7": bool(flags.get("seed_pair_layer7", False)),
        "seed_prefilter": bool(flags.get("seed_prefilter", False)),
        "seed_delta_s": bool(flags.get("seed_delta_s", False)),
        "seed_delta_w": bool(flags.get("seed_delta_w", False)),
        "seed_forward": bool(flags.get("seed_forward", False)),
        "inward_walk": bool(flags.get("inward_walk", False)),
        "final_delta_s": final_ds,
        "hit_pattern": pattern,
    }


@dataclass
class CutFlow:
    """Sequential pass counts. An event is counted at a cut only if every earlier cut passed."""

    names: tuple[str, ...] = tuple(name for name, _definition in CUT_FLOW)
    counts: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.counts = {name: 0 for name in self.names}

    def add(self, record: dict[str, bool]) -> bool:
        alive = True
        for name in self.names:
            alive = alive and bool(record.get(name, False))
            if alive:
                self.counts[name] += 1
        return alive

    def rows(self) -> list[dict]:
        previous = None
        table = []
        definitions = dict(CUT_FLOW)
        for name in self.names:
            count = self.counts[name]
            if previous is None:
                conditional = 1.0 if count else 0.0
            elif previous == 0:
                conditional = 0.0
            else:
                conditional = count / previous
            table.append(
                {
                    "cut": name,
                    "definition": definitions[name],
                    "n_pass": count,
                    "efficiency_given_previous": conditional,
                }
            )
            previous = count
        return table

    def as_dict(self) -> dict:
        rows = self.rows()
        n_gen = self.counts["generated"]
        n_trig = self.counts["trigger"]
        n_fid = self.counts["fiducial"]
        n_reco = self.counts["hit_pattern"]
        return {
            "definitions": {
                "epsilon_trigger": "N(trigger) / N(generated)",
                "epsilon_fiducial": "N(fiducial) / N(trigger)",
                "epsilon_reco": "N(hit_pattern) / N(fiducial)",
                "epsilon_total": "epsilon_trigger * epsilon_fiducial * epsilon_reco",
            },
            "epsilon_trigger": (n_trig / n_gen) if n_gen else 0.0,
            "epsilon_fiducial": (n_fid / n_trig) if n_trig else 0.0,
            "epsilon_reco": (n_reco / n_fid) if n_fid else 0.0,
            "epsilon_total": ((n_trig / n_gen) * (n_fid / n_trig) * (n_reco / n_fid)) if n_fid else 0.0,
            "cuts": rows,
        }


def classify_event(event: SignalEvent, cuts: PaperCuts) -> tuple[PlaneResult, bool]:
    result = find_plane(event.measured.reconstructed, cuts)
    reconstructed = passes_signal_region(result, cuts)
    return result, reconstructed


def accumulate_event(stats: EfficiencyStats, event: SignalEvent, cuts: PaperCuts) -> PlaneResult:
    result, reconstructed = classify_event(event, cuts)
    stats.add(
        triggered=event.triggered,
        fiducial=event.measured.fiducial(cuts.n_layers),
        reconstructed=reconstructed,
    )
    return result


@dataclass
class SampleResult:
    stats: EfficiencyStats
    cutflow: CutFlow
    events: list[dict]


def run_sample(
    detector: Detector,
    cuts: PaperCuts,
    mass_gev: float,
    lambda_ev: float,
    mode: str,
    n_events: int,
    seed: int,
    progress_every: int = 0,
) -> SampleResult:
    rng = np.random.default_rng(seed)
    stats = EfficiencyStats()
    cutflow = CutFlow()
    events: list[dict] = []
    for index in range(n_events):
        event = simulate_signal_event(
            rng,
            detector,
            mass_gev=mass_gev,
            lambda_ev=lambda_ev,
            mode=mode,
            pt_trigger_gev=cuts.pt_trigger_gev,
        )
        result = accumulate_event(stats, event, cuts)
        record = event_cut_record(event, result, cuts)
        cutflow.add(record)
        events.append(
            {
                "event": index,
                "pair_pt_gev": event.pair_pt_gev,
                "delta_s_mm": None if result.delta_s_mm == np.inf else result.delta_s_mm,
                "delta_w_mm": None if result.delta_w_mm == np.inf else result.delta_w_mm,
                **{name: int(record[name]) for name in cutflow.names},
            }
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"{mode}: {index + 1}/{n_events}", flush=True)
    return SampleResult(stats=stats, cutflow=cutflow, events=events)


def _event_row(index: int, event: SignalEvent, result: PlaneResult, record: dict[str, bool], names: tuple[str, ...]) -> dict:
    return {
        "event": index,
        "pair_pt_gev": event.pair_pt_gev,
        "delta_s_mm": None if result.delta_s_mm == np.inf else result.delta_s_mm,
        "delta_w_mm": None if result.delta_w_mm == np.inf else result.delta_w_mm,
        **{name: int(record[name]) for name in names},
    }


def _result_from_rows(rows: list[dict]) -> SampleResult:
    stats = EfficiencyStats()
    cutflow = CutFlow()
    ordered = sorted(rows, key=lambda row: row["event"])
    for row in ordered:
        record = {name: bool(row[name]) for name in cutflow.names}
        survived = cutflow.add(record)
        stats.add(
            triggered=record["trigger"],
            fiducial=record["fiducial"],
            reconstructed=survived,
        )
    return SampleResult(stats=stats, cutflow=cutflow, events=ordered)


def _parallel_event_batch(spec: dict) -> list[dict]:
    """Simulate one independent batch. Each event has its own seed."""
    detector, cuts = load_detector_and_cuts()
    names = tuple(name for name, _definition in CUT_FLOW)
    rows = []
    for index in spec["indices"]:
        rng = np.random.default_rng(int(spec["base_seed"]) + int(index) * 1_000_003)
        event = simulate_signal_event(
            rng,
            detector,
            mass_gev=float(spec["mass_gev"]),
            lambda_ev=float(spec["lambda_ev"]),
            mode=str(spec["mode"]),
            pt_trigger_gev=cuts.pt_trigger_gev,
        )
        result = accumulate_event(EfficiencyStats(), event, cuts)
        record = event_cut_record(event, result, cuts)
        rows.append(_event_row(int(index), event, result, record, names))
    return rows


def run_sample_parallel(
    mass_gev: float,
    lambda_ev: float,
    mode: str,
    n_events: int,
    seed: int,
    workers: int = 16,
) -> SampleResult:
    """Split events across processes. One event never shares a core with another."""
    n_workers = max(1, min(int(workers), int(n_events)))
    chunks: list[list[int]] = [[] for _ in range(n_workers)]
    for index in range(n_events):
        chunks[index % n_workers].append(index)
    specs = [
        {
            "indices": chunk,
            "base_seed": int(seed),
            "mass_gev": float(mass_gev),
            "lambda_ev": float(lambda_ev),
            "mode": str(mode),
        }
        for chunk in chunks
        if chunk
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_parallel_event_batch, spec) for spec in specs]
        finished = 0
        for future in futures:
            batch = future.result()
            rows.extend(batch)
            finished += len(batch)
            print(f"{mode}: {finished}/{n_events}", flush=True)
    return _result_from_rows(rows)


def load_detector_and_cuts():
    return detector_from_config(), cuts_from_config()
