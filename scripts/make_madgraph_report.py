#!/usr/bin/env python3
"""Build the MadGraph/Pythia cut-flow report and Figures 2-5."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search.detector import default_detector, measure_pair
from quirk_search.dynamics import PairState, propagate_pair
from quirk_search.madgraph_events import parse_hepmc
from quirk_search.plane_finder import PaperCuts, find_plane

_SPEC = importlib.util.spec_from_file_location("make_report", ROOT / "scripts" / "make_report.py")
_REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORT)

FIGURE_DIR = ROOT / "results" / "report" / "madgraph_figures"
REPORT_DIR = ROOT / "results" / "report"
MG_DIR = ROOT / "results" / "madgraph"

# Matched cross sections printed by MadGraph for the runs that were analysed.
# The JSON field cross_section_pb is the older toy estimate and is not used here.
CROSS_SECTIONS = {
    "ew_800": {
        "parton_pb": 0.0003814,
        "parton_err_pb": 1.901e-6,
        "matched_pb": 0.0001983,
        "matched_err_pb": 4.374e-6,
        "requested": 2000,
        "matched": 1040,
    },
    "qcd_1800": {
        "parton_pb": 0.0007566,
        "parton_err_pb": 3.7e-6,
        "matched_pb": 0.0003234,
        "matched_err_pb": 8.518e-6,
        "requested": 2000,
        "matched": 855,
    },
}

PAPER = {
    "ew_800": {"trigger": 0.10, "fiducial": 0.28, "reco": 0.72},
    "qcd_1800": {"trigger": 0.24, "fiducial": 0.28, "reco": None},
}

CUT_LABELS = [
    ("generated", "Generated"),
    ("trigger", "Trigger"),
    ("fiducial", "Fiducial"),
    ("seed_pair_layer8", "Layer-8 seed"),
    ("seed_pair_layer7", "Layer-7 seed"),
    ("seed_prefilter", "Seed prefilter"),
    ("seed_delta_s", r"Seed \(\Delta s\)"),
    ("seed_delta_w", r"Seed \(\Delta w\)"),
    ("seed_forward", "Forward"),
    ("inward_walk", "Inward walk"),
    ("final_delta_s", r"Final \(\Delta s\)"),
    ("hit_pattern", "Hit pattern"),
]


def fmt(x: float) -> str:
    return f"{x:.3f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def sci_xsec(x: float) -> str:
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    return rf"{mant:.3f}\times 10^{{{exp}}}"


def sci(x: float) -> str:
    ax = abs(x)
    if ax == 0.0 or not math.isfinite(x):
        return "0"
    exp = int(math.floor(math.log10(ax)))
    mant = x / 10**exp
    return rf"{mant:.2f}\times 10^{{{exp}}}"


def cut_rows(summary: dict) -> list[dict]:
    rows = []
    previous = None
    for cut in summary["cutflow"]["cuts"]:
        k = int(cut["n_pass"])
        if cut["cut"] == "generated":
            item = {"cut": cut["cut"], "k": k, "n": k, "p": 1.0, "se": 0.0, "low": 1.0, "high": 1.0}
        else:
            n = int(previous)
            p, se, low, high = _REPORT.wilson(k, n)
            item = {"cut": cut["cut"], "k": k, "n": n, "p": p, "se": se, "low": low, "high": high}
        rows.append(item)
        previous = k
    return rows


def efficiencies(summary: dict) -> dict:
    counts = {
        "trigger": (int(summary["n_triggered"]), int(summary["n_generated"])),
        "fiducial": (int(summary["n_fiducial"]), int(summary["n_triggered"])),
        "reco": (int(summary["n_reconstructed"]), int(summary["n_fiducial"])),
        "total": (int(summary["n_reconstructed"]), int(summary["n_generated"])),
    }
    out = {}
    for name, (k, n) in counts.items():
        p, se, low, high = _REPORT.wilson(k, n)
        out[name] = {"k": k, "n": n, "p": p, "se": se, "low": low, "high": high}
    return out


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_widths(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ds, dw = [], []
    for row in load_rows(path):
        if row["trigger"] != "1" or row["fiducial"] != "1":
            continue
        if row["delta_s_mm"] == "" or row["delta_w_mm"] == "":
            continue
        ds.append(float(row["delta_s_mm"]) / 10.0)
        dw.append(float(row["delta_w_mm"]) / 10.0)
    return np.asarray(ds), np.asarray(dw)


def pick_event(rows: list[dict]) -> dict:
    good = [row for row in rows if row["hit_pattern"] == "1" and row["delta_s_mm"] != ""]
    if not good:
        raise RuntimeError("no reconstructed event to draw")
    pts = sorted(float(row["pair_pt_gev"]) for row in good)
    target = pts[len(pts) // 2]
    return min(good, key=lambda row: abs(float(row["pair_pt_gev"]) - target))


def propagate_record(record: dict, mass_gev: float):
    detector = default_detector()
    state = PairState(
        x1=np.array(record["x1"], dtype=float),
        x2=np.array(record["x2"], dtype=float),
        p1=np.array(record["p1"], dtype=float),
        p2=np.array(record["p2"], dtype=float),
        mass_gev=mass_gev,
        lambda_ev=4000.0,
        charge1=float(record["charge1"]),
        charge2=float(record["charge2"]),
    )
    trajectory = propagate_pair(
        state,
        b_field_t=detector.b_field_t,
        t_max_ns=detector.timing_ns,
        outer_radius_mm=detector.outer_radius_mm,
    )
    return detector, trajectory


def stride(curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(curves["r_cm_mm"])
    step = max(1, n // 4000)
    return {key: value[::step] for key, value in curves.items()}


def figure_dynamics(ew: dict, qcd: dict) -> None:
    ew = stride(ew)
    qcd = stride(qcd)
    fig, ax_l = plt.subplots(figsize=(7.2, 4.6))
    ax_d = ax_l.twinx()
    ax_l.plot(ew["r_cm_mm"], ew["l_over_hbar"], color="#1f4e79", lw=1.2, label=r"$|L|/\hbar$")
    ax_d.plot(ew["r_cm_mm"], ew["separation_mm"], color="#c47b2b", lw=1.2, label="separation $d$")
    ax_l.set_xlabel("Transverse radius of the quirk-pair centre of mass [mm]")
    ax_l.set_ylabel(r"Internal angular momentum $|L|/\hbar$")
    ax_d.set_ylabel("Quirk separation $d$ [mm]")
    ax_l.set_title("Angular momentum and separation")
    lines = ax_l.get_lines() + ax_d.get_lines()
    ax_l.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_angular_momentum.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ew["r_cm_mm"], ew["delta_phi"], color="#1f4e79", lw=1.2, label="800 GeV electroweak")
    ax.plot(qcd["r_cm_mm"], qcd["delta_phi"], color="#b85c38", lw=1.2, label="1800 GeV colored")
    ax.set_xlabel("Transverse radius of the quirk-pair centre of mass [mm]")
    ax.set_ylabel(r"Plane rotation $\Delta\phi$ [rad]")
    ax.set_title("Rotation of the quirk plane in a 2 T field")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_plane_rotation.png", dpi=160)
    plt.close(fig)


def figure_plane(detector, trajectory, seed: int) -> dict:
    measured = measure_pair(detector, trajectory, np.random.default_rng(seed))
    result = find_plane(measured.reconstructed, PaperCuts())
    if not result.found:
        raise RuntimeError("selected electroweak event did not yield a plane")
    points = np.array([[hit.x, hit.y, hit.z] for hit in measured.reconstructed], dtype=float)
    selected = points[result.indices]
    long_coord = selected @ result.n3
    width_coord = selected @ result.n2
    radii = [layer.radius_mm for layer in detector.layers]
    ellipses = _REPORT.project_layers(result.n3, result.n2, radii)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for ellipse in ellipses:
        ax.plot(ellipse[:, 0], ellipse[:, 1], color="#9aa3ad", lw=0.8, ls=":")
    ax.scatter(long_coord, width_coord, s=18, color="#1f4e79", zorder=3, label="quirk hits on the plane")
    ax.set_xlabel(r"Coordinate along $n_3$ [mm]")
    ax.set_ylabel(r"Coordinate along $n_2$ [mm]")
    ax.set_title("Hits projected onto the reconstructed plane")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, loc="best")
    inset = ax.inset_axes([0.15, 0.55, 0.36, 0.36])
    inset.scatter(long_coord, width_coord, s=16, color="#1f4e79")
    inset.set_xlim(float(np.min(long_coord) - 5.0), float(np.max(long_coord) + 5.0))
    pad = max(1.0, 3.0 * result.delta_w_mm)
    inset.set_ylim(float(np.min(width_coord) - pad), float(np.max(width_coord) + pad))
    inset.set_title(rf"$\Delta w = {result.delta_w_mm / 10.0:.3f}$ cm", fontsize=8)
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_plane_projection.png", dpi=160)
    plt.close(fig)
    return {
        "delta_s_cm": result.delta_s_mm / 10.0,
        "delta_w_cm": result.delta_w_mm / 10.0,
        "n_hits": int(len(result.indices)),
    }


def figure_widths(samples: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    colors = {"ew_800": "#1f4e79", "qcd_1800": "#b85c38"}
    labels = {"ew_800": "Electroweak, 800 GeV", "qcd_1800": "Colored, 1800 GeV"}
    for key, (ds, dw) in samples.items():
        axes[0].hist(ds, bins=16, range=(0.0, 0.02), histtype="step", lw=1.6, color=colors[key], label=f"{labels[key]} (N={len(ds)})")
        axes[1].hist(dw, bins=16, range=(0.0, 1.2), histtype="step", lw=1.6, color=colors[key], label=labels[key])
    axes[0].axvline(0.01, color="#333333", ls="--", lw=1.0, label=r"signal cut $\Delta s < 0.01$ cm")
    axes[1].axvline(1.0, color="#333333", ls="--", lw=1.0, label=r"width cut $\Delta w < 1$ cm")
    axes[0].set_xlabel(r"Plane thickness $\Delta s$ [cm]")
    axes[1].set_xlabel(r"Oscillation width $\Delta w$ [cm]")
    for axis in axes:
        axis.set_ylabel("Fiducial events with a reconstructed plane")
    axes[0].set_title(r"$\Delta s$")
    axes[1].set_title(r"$\Delta w$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig5_delta_s_delta_w.png", dpi=160)
    plt.close(fig)


def cut_table(rows: list[dict]) -> str:
    body = []
    for (key, label), row in zip(CUT_LABELS, rows):
        if row["cut"] != key:
            raise RuntimeError(f"unexpected cut {row['cut']}")
        body.append(
            f"{label} & {row['k']} & {row['n']} & {fmt(row['p'])} & {fmt(row['se'])} & {fmt_ci(row['low'], row['high'])} \\\\"
        )
    return "\n".join(body)


def eff_table(eff: dict) -> str:
    lines = []
    for key, label in (
        ("trigger", r"\(\varepsilon_{\mathrm{trig}}\)"),
        ("fiducial", r"\(\varepsilon_{\mathrm{fid}}\)"),
        ("reco", r"\(\varepsilon_{\mathrm{reco}}\)"),
        ("total", r"\(\varepsilon\)"),
    ):
        item = eff[key]
        lines.append(
            f"{label} & ${item['k']}/{item['n']}$ & {fmt(item['p'])} & {fmt(item['se'])} & {fmt_ci(item['low'], item['high'])} \\\\"
        )
    return "\n".join(lines)


def verdict(paper: float | None, item: dict) -> str:
    if paper is None:
        return "not tabulated at 4 keV"
    return "inside" if item["low"] <= paper <= item["high"] else "outside"


def comparison_rows(effs: dict[str, dict]) -> str:
    specs = [
        ("ew_800", "Electroweak", "trigger", "0.10"),
        ("ew_800", "Electroweak", "fiducial", "0.28"),
        ("ew_800", "Electroweak", "reco", "0.72"),
        ("qcd_1800", "Colored", "trigger", "0.24"),
        ("qcd_1800", "Colored", "fiducial", "0.28"),
        ("qcd_1800", "Colored", "reco", r"0.59 at \SI{3}{\keV}; 0.74 at \SI{5}{\keV}"),
    ]
    symbols = {"trigger": r"\(\varepsilon_{\mathrm{trig}}\)", "fiducial": r"\(\varepsilon_{\mathrm{fid}}\)", "reco": r"\(\varepsilon_{\mathrm{reco}}\)"}
    lines = []
    for sample, label, key, paper_text in specs:
        item = effs[sample][key]
        paper = PAPER[sample][{"trigger": "trigger", "fiducial": "fiducial", "reco": "reco"}[key]]
        lines.append(
            f"{label} & {symbols[key]} & {paper_text} & {fmt(item['p'])} & {fmt(item['se'])} & {fmt_ci(item['low'], item['high'])} & {verdict(paper, item)} \\\\"
        )
    return "\n".join(lines)


def curve_notes(curves: dict[str, np.ndarray]) -> dict[str, float]:
    finite_d = curves["separation_mm"][np.isfinite(curves["separation_mm"])]
    return {
        "l_max": float(np.nanmax(curves["l_over_hbar"])),
        "d_min": float(np.min(finite_d)),
        "d_max": float(np.max(finite_d)),
        "phi_max": float(np.nanmax(curves["delta_phi"])),
    }


def yield_count(matched_pb: float, epsilon: float, fraction: float) -> float:
    return matched_pb * 300.0 * 1000.0 * epsilon * fraction


def write_tex(ctx: dict) -> None:
    ew_rows = cut_table(ctx["rows"]["ew_800"])
    qcd_rows = cut_table(ctx["rows"]["qcd_1800"])
    ew_eff = eff_table(ctx["eff"]["ew_800"])
    qcd_eff = eff_table(ctx["eff"]["qcd_1800"])
    comp = comparison_rows(ctx["eff"])
    ew = ctx["notes"]["ew"]
    qcd = ctx["notes"]["qcd"]
    plane = ctx["plane"]
    n_ew = ctx["n_widths"]["ew_800"]
    n_qcd = ctx["n_widths"]["qcd_1800"]
    y_ew = yield_count(CROSS_SECTIONS["ew_800"]["matched_pb"], ctx["eff"]["ew_800"]["total"]["p"], 1.0)
    y_qcd = yield_count(CROSS_SECTIONS["qcd_1800"]["matched_pb"], ctx["eff"]["qcd_1800"]["total"]["p"], 0.30)
    text = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{siunitx}}
\DeclareSIUnit{{\GeV}}{{\giga\electronvolt}}
\DeclareSIUnit{{\TeV}}{{\tera\electronvolt}}
\DeclareSIUnit{{\keV}}{{\kilo\electronvolt}}
\DeclareSIUnit{{\ns}}{{\nano\second}}
\DeclareSIUnit{{\cm}}{{\centi\metre}}
\DeclareSIUnit{{\mm}}{{\milli\metre}}
\DeclareSIUnit{{\rad}}{{\radian}}
\DeclareSIUnit{{\pb}}{{pb}}
\DeclareSIUnit{{\fb}}{{fb}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{microtype}}
\usepackage{{setspace}}
\setstretch{{1.08}}

\graphicspath{{{{madgraph_figures/}}}}

\title{{Quirk plane search with MadGraph and Pythia:\\ efficiencies, uncertainties, and Figures 2--5}}
\author{{Comparison with Knapen, Lou, Papucci, and Setford, arXiv:1708.02243}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This note reports the MadGraph5\_aMC@NLO and Pythia~8 samples of the quirk plane search.
The electroweak sample is Drell--Yan production of an \SI{{800}}{{\GeV}} vector-like lepton.
The colored sample is QCD production of an \SI{{1800}}{{\GeV}} vector-like quark.
Both use a string scale of \SI{{4}}{{\keV}}.
Every cut is counted sequentially.
Each efficiency is given with its binomial standard error and a Wilson 95\% confidence interval, and is compared with Table~I of the paper.
The generator settings are listed in full.
Figures~2--5 of the paper are reproduced for these benchmarks: the internal angular momentum and separation, the rotation of the quirk plane, one event projected onto its reconstructed plane, and the distributions of the thickness \(\Delta s\) and the width \(\Delta w\).
\end{{abstract}}

\section{{Generator settings}}
The hard process and the parton shower come from MadGraph5\_aMC@NLO~3.8.0 and Pythia~8.317.
The quirk oscillation is not part of Pythia.
After the shower, each matched event is propagated with the straight-string equation of motion in a \SI{{2}}{{\tesla}} solenoid, at \(\Lambda = \SI{{4}}{{\keV}}\), and passed through the eight-layer barrel and the plane finder.

\subsection{{Collider and run card}}
Both processes use the same collider settings.
\begin{{center}}
\begin{{tabular}}{{ll}}
\toprule
Setting & Value \\
\midrule
Beams & proton--proton \\
Beam energy & \SI{{6500}}{{\GeV}} each, so \(\sqrt{{s}} = \SI{{13}}{{\TeV}}\) \\
PDF & NNPDF2.3 LO, LHAPDF set \texttt{{nn23lo1}} \\
Random seed & 170802243 \\
Events requested & 2000 unweighted events \\
Cores & 16 \\
Extra-jet matching & MLM, \texttt{{ickkw = 1}} \\
Matching scale \(Q_{{\mathrm{{cut}}}}\) & \texttt{{xqcut = 30}}~\si{{\GeV}} \\
Pythia jet-matching scale & \texttt{{JetMatching:qCut = 45}}~\si{{\GeV}} \\
Maximum extra jets & \texttt{{JetMatching:nJetMax = 1}} \\
Light flavours in the match & \texttt{{JetMatching:nQmatch = 4}} \\
Jet algorithm & \(k_T\), exclusive, cone radius 1, \texttt{{doShowerKt = off}} \\
Jet \(p_T\) cut in the matrix element & \texttt{{ptj = 10}}~\si{{\GeV}} \\
Scale and PDF systematics & off (\texttt{{use\_syst = False}}) \\
MadSpin & off \\
Analysis & off \\
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection{{Electroweak process}}
The electroweak quirk is the charge\(-1\) vector-like lepton \texttt{{taup}} in the model \texttt{{VLLS\_NLO}}.
Its PDG code is 17 and it is a color singlet.
The mass parameter is \(\texttt{{MTAUP}} = \SI{{800}}{{\GeV}}\).
The width is set to zero by
\(\texttt{{epstaupeL}} = \texttt{{epstaupmuL}} = \texttt{{epstauptaL}} = 0\).
The generated processes are
\begin{{align*}}
p p &\to \tau'\bar{{\tau}}', && \texttt{{QED=2}}, \\
p p &\to \tau'\bar{{\tau}}'\, j, && \texttt{{QED=2}},\ \texttt{{QCD=1}}.
\end{{align*}}
Hadronization is left at the Pythia default.
The lepton is a color singlet, so there is no QCD string between the pair.
MadGraph reports a parton-level cross section of
\({sci_xsec(CROSS_SECTIONS['ew_800']['parton_pb'])} \pm {sci_xsec(CROSS_SECTIONS['ew_800']['parton_err_pb'])}\)~\si{{\pb}}
for 2000 events.
MLM matching keeps 1040 events.
The matched cross section is
\({sci_xsec(CROSS_SECTIONS['ew_800']['matched_pb'])} \pm {sci_xsec(CROSS_SECTIONS['ew_800']['matched_err_pb'])}\)~\si{{\pb}}.
The efficiencies below use these 1040 matched events as \(N_{{\mathrm{{gen}}}}\).

\subsection{{Colored process}}
The colored quirk is the charge \(+2/3\) vector-like quark \texttt{{tp}} in the model \texttt{{VLQ\_UFO}}.
Its PDG code is 6000006 and it is a color triplet.
The mass parameter is \(\texttt{{MTP}} = \SI{{1800}}{{\GeV}}\) and the width is \(\texttt{{WTP}} = 0\).
The generated processes are
\begin{{align*}}
p p &\to t' \bar t', \\
p p &\to t' \bar t' j.
\end{{align*}}
Pythia hadronization is switched off with \texttt{{HadronLevel:all = off}}.
A stable colored pair does not leave enough energy for a QCD string to fragment, and with hadronization left on the shower never writes an event.
The dark string is the straight-string propagator in this package, not the Pythia string.
The parton shower and the MLM match are still performed.
MadGraph reports a parton-level cross section of
\({sci_xsec(CROSS_SECTIONS['qcd_1800']['parton_pb'])} \pm {sci_xsec(CROSS_SECTIONS['qcd_1800']['parton_err_pb'])}\)~\si{{\pb}}
for 2000 events.
MLM matching keeps 855 events.
The matched cross section is
\({sci_xsec(CROSS_SECTIONS['qcd_1800']['matched_pb'])} \pm {sci_xsec(CROSS_SECTIONS['qcd_1800']['matched_err_pb'])}\)~\si{{\pb}}.
The efficiencies below use these 855 matched events as \(N_{{\mathrm{{gen}}}}\).

These two models are vector-like fermions used as proxies for the quirk.
Their cross sections are the MadGraph results for those models.
They are not the quirk cross sections used for the discovery contours in the paper.
The JSON files also contain a field \texttt{{cross\_section\_pb}} from an earlier toy estimate.
That field is not the MadGraph cross section and is not used in this note.

\subsection{{Quirk propagation and the detector}}
Each matched event supplies the two heavy-fermion three-momenta.
The pair is started at the origin with a separation of \(10^{{-3}}\)~\si{{\mm}} along the relative velocity.
The straight-string force and the Lorentz force are integrated in a \SI{{2}}{{\tesla}} field along \(z\), until both quirks leave the outer barrel or \SI{{25}}{{\ns}} have passed.
Hits are measured relative to that vertex.
The \SI{{45}}{{\mm}} beam spot is not sampled again on top of the MadGraph event.
Pixel hits are moved to the pixel centre: \SI{{50}}{{\micro\metre}} in \(r\phi\), \SI{{0.25}}{{\mm}} in \(z\) on the innermost layer and \SI{{0.40}}{{\mm}} on the other pixel layers.
SCT hits are smeared by \SI{{34}}{{\micro\metre}} in \(r\phi\), twice the \SI{{17}}{{\micro\metre}} intrinsic resolution.
The SCT stereo \(z\) uncertainty is about \SI{{1}}{{\mm}}.
It is not applied as a shift of the hit, because it is larger than the \SI{{0.1}}{{\mm}} thickness cut.
Neighboring hits are merged.
No pileup and no photon conversions are overlaid on these samples.
The colored charged-hadron fraction of 30\% is not applied inside the efficiency.
It is used only when a yield is quoted.

\section{{Efficiency definitions}}
An event is counted at a cut only if it has passed every earlier cut.
The three efficiencies in the paper are ratios of those counts,
\begin{{align}}
\varepsilon_{{\mathrm{{trig}}}}
  &= \frac{{N_{{\mathrm{{trig}}}}}}{{N_{{\mathrm{{gen}}}}}},
  &
  N_{{\mathrm{{trig}}}}
  &= \#\{{p_T > \SI{{200}}{{\GeV}}\}},
  \label{{eq:trig}}
  \\
\varepsilon_{{\mathrm{{fid}}}}
  &= \frac{{N_{{\mathrm{{fid}}}}}}{{N_{{\mathrm{{trig}}}}}},
  \label{{eq:fid}}
  \\
\varepsilon_{{\mathrm{{reco}}}}
  &= \frac{{N_{{\mathrm{{reco}}}}}}{{N_{{\mathrm{{fid}}}}}},
  \label{{eq:reco}}
  \\
\varepsilon
  &= \varepsilon_{{\mathrm{{trig}}}}\,
     \varepsilon_{{\mathrm{{fid}}}}\,
     \varepsilon_{{\mathrm{{reco}}}}
   = \frac{{N_{{\mathrm{{reco}}}}}}{{N_{{\mathrm{{gen}}}}}}.
  \label{{eq:total}}
\end{{align}}
Here \(N_{{\mathrm{{gen}}}}\) is the number of events kept by MLM matching, 1040 for the electroweak sample and 855 for the colored sample.
The fiducial count \(N_{{\mathrm{{fid}}}}\) requires both quirks to intersect all eight barrel layers, with every hit time below \SI{{25}}{{\ns}}.
The reconstruction count \(N_{{\mathrm{{reco}}}}\) is the number of events that pass the final hit-pattern cut.
Because the flow is sequential, that count already requires the trigger, the fiducial requirement, the seed windows, the factor-of-three growth test, \(\Delta w < \SI{{1}}{{\cm}}\), and \(\Delta s < \SI{{0.01}}{{\cm}}\).

The individual cuts, in order, are:
\begin{{enumerate}}
\item \textbf{{Generated.}} Every matched event.
\item \textbf{{Trigger.}} Pair transverse momentum above \SI{{200}}{{\GeV}}.
\item \textbf{{Fiducial.}} Both quirks hit all eight layers within \SI{{25}}{{\ns}}.
\item \textbf{{Layer-8 seed.}} A hit pair on layer 8 with \(\Delta\phi < 0.1\) and \(\Delta z < \SI{{2}}{{\cm}}\).
\item \textbf{{Layer-7 seed.}} A hit pair on layer 7 in the same window.
\item \textbf{{Seed prefilter.}} The two pairs agree within \SI{{0.5}}{{\rad}} and \SI{{80}}{{\mm}}. This removes combinations that cannot satisfy the \SI{{1}}{{\cm}} width cut.
\item \textbf{{Seed thickness.}} The four seed hits have \(\Delta s < \SI{{0.05}}{{\cm}}\).
\item \textbf{{Seed width.}} The four seed hits have \(\Delta w < \SI{{1}}{{\cm}}\).
\item \textbf{{Forward direction.}} Every seed hit satisfies \(\boldsymbol{{n}}_3\cdot\boldsymbol{{x}} > 0\).
\item \textbf{{Inward walk.}} From layer 6 down to layer 1, a hit is added only when both \(\Delta s\) and \(\Delta w\) stay within a factor of three of their current values, and the finished plane has \(\Delta w < \SI{{1}}{{\cm}}\).
\item \textbf{{Final thickness.}} The finished plane has \(\Delta s < \SI{{0.01}}{{\cm}}\).
\item \textbf{{Hit pattern.}} Seven layers contain two hits and the remaining layer contains at least one.
\end{{enumerate}}

\section{{Wilson intervals}}
Each efficiency is a binomial proportion \(k/n\), with estimate \(\hat{{p}} = k/n\).
The standard error is
\begin{{equation}}
\sigma = \sqrt{{\frac{{\hat{{p}}\,(1-\hat{{p}})}}{{n}}}}.
\label{{eq:se}}
\end{{equation}}
The 95\% interval is the Wilson score interval. With \(z = 1.960\),
\begin{{equation}}
\begin{{aligned}}
\hat{{p}}_{{\mathrm{{W}}}}
&=
\left(\hat{{p}} + \frac{{z^{{2}}}}{{2n}}\right)
\frac{{1}}{{1 + z^{{2}}/n}},
\\
\delta
&=
\frac{{z}}{{1 + z^{{2}}/n}}
\sqrt{{
  \frac{{\hat{{p}}\,(1-\hat{{p}})}}{{n}}
  + \frac{{z^{{2}}}}{{4n^{{2}}}}
}},
\end{{aligned}}
\label{{eq:wilson}}
\end{{equation}}
and the interval is \(\hat{{p}}_{{\mathrm{{W}}}} \pm \delta\), clipped to \([0,1]\).
For a conditional cut, \(k\) is the number of events that pass the cut and \(n\) is the number that reached it.
The paper does not quote uncertainties on Table~I.

\section{{Electroweak sample}}
\label{{sec:ew}}
Mass \SI{{800}}{{\GeV}}, string scale \SI{{4}}{{\keV}}, \(N_{{\mathrm{{gen}}}} = 1040\) matched events.

\begin{{table}}[H]
\centering
\caption{{Sequential cut flow for the electroweak MadGraph sample. The efficiency is \(k/n\), where \(n\) is the number of events that reached the cut.}}
\label{{tab:ew-cuts}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Cut & \(N_{{\mathrm{{pass}}}}\) & \(N_{{\mathrm{{in}}}}\) & Efficiency & Std.\ error & Wilson 95\% CI \\
\midrule
{ew_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Electroweak efficiencies as defined in Eqs.~\eqref{{eq:trig}}--\eqref{{eq:total}}.}}
\label{{tab:ew-eff}}
\begin{{tabular}}{{lrrrr}}
\toprule
Efficiency & Count & Estimate & Std.\ error & Wilson 95\% CI \\
\midrule
{ew_eff}
\bottomrule
\end{{tabular}}
\end{{table}}

The last row of Table~\ref{{tab:ew-cuts}} is not \(\varepsilon_{{\mathrm{{reco}}}}\).
It is the hit-pattern efficiency among events that already passed the final \(\Delta s\) cut.
The paper's reconstruction efficiency uses the fiducial sample as the denominator.

\section{{Colored sample}}
\label{{sec:qcd}}
Mass \SI{{1800}}{{\GeV}}, string scale \SI{{4}}{{\keV}}, \(N_{{\mathrm{{gen}}}} = 855\) matched events.

\begin{{table}}[H]
\centering
\caption{{Sequential cut flow for the colored MadGraph sample.}}
\label{{tab:qcd-cuts}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Cut & \(N_{{\mathrm{{pass}}}}\) & \(N_{{\mathrm{{in}}}}\) & Efficiency & Std.\ error & Wilson 95\% CI \\
\midrule
{qcd_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Colored efficiencies as defined in Eqs.~\eqref{{eq:trig}}--\eqref{{eq:total}}.}}
\label{{tab:qcd-eff}}
\begin{{tabular}}{{lrrrr}}
\toprule
Efficiency & Count & Estimate & Std.\ error & Wilson 95\% CI \\
\midrule
{qcd_eff}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Comparison with Table~I}}
Table~I of arXiv:1708.02243 gives \(\varepsilon_{{\mathrm{{trig}}}} = 0.10\) and \(\varepsilon_{{\mathrm{{fid}}}} = 0.28\) for the \SI{{800}}{{\GeV}} electroweak benchmark, independent of \(\Lambda\), and \(\varepsilon_{{\mathrm{{reco}}}} = 0.72\) at \(\Lambda = \SI{{4}}{{\keV}}\).
For the \SI{{1800}}{{\GeV}} colored benchmark it gives \(\varepsilon_{{\mathrm{{trig}}}} = 0.24\) and \(\varepsilon_{{\mathrm{{fid}}}} = 0.28\).
It does not list a colored reconstruction efficiency at \SI{{4}}{{\keV}}.
The neighboring colored entries are \(\varepsilon_{{\mathrm{{reco}}}} = 0.59\) at \SI{{3}}{{\keV}} and \(\varepsilon_{{\mathrm{{reco}}}} = 0.74\) at \SI{{5}}{{\keV}}.

\begin{{table}}[H]
\centering
\caption{{Table~I compared with the MadGraph samples. A Table~I value is marked outside when it does not lie in the Wilson 95\% interval.}}
\label{{tab:compare}}
\footnotesize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llp{{4.6cm}}rrll}}
\toprule
Sample & Efficiency & Table~I & This work & Std.\ error & Wilson 95\% CI & Interval \\
\midrule
{comp}
\bottomrule
\end{{tabular}}}}
\end{{table}}

The colored trigger efficiency and both fiducial efficiencies agree with Table~I: each Table~I value lies inside the Wilson interval.
The electroweak trigger efficiency is higher than 0.10, and 0.10 lies outside the interval.
The vector-like lepton model, with one extra jet and MLM matching, is not the same matrix element as the quirk model used for Table~I, so the momentum distribution is not expected to match that entry exactly.
The electroweak reconstruction efficiency is \(22/58\).
The Table~I value 0.72 lies outside its Wilson interval.
The colored reconstruction efficiency at \SI{{4}}{{\keV}} is not in Table~I.
Its central value, 0.537, lies below the paper's \SI{{3}}{{\keV}} entry of 0.59, and 0.59 lies inside the Wilson interval.
The \SI{{5}}{{\keV}} entry 0.74 lies outside that interval.
The fiducial samples that enter \(\varepsilon_{{\mathrm{{reco}}}}\) contain 58 and 54 events, so this efficiency is the least precise of the three.

At \SI{{300}}{{\fb^{{-1}}}}, the central matched cross section and the central total efficiency give about \({y_ew:.2f}\) electroweak events.
For the colored sample the same product, multiplied by the 30\% charged fraction, gives about \({y_qcd:.2f}\) events.
These yields use the vector-like-fermion cross sections above.
They are not the quirk yields of the paper.

\section{{Figures}}
\setcounter{{figure}}{{1}}
The four figures below correspond to Figures~2--5 of the paper.
Figures~2 and~3 integrate the straight-string equation at the two benchmark masses, \(\Lambda = \SI{{4}}{{\keV}}\), in the \SI{{2}}{{\tesla}} field.
A showered MadGraph event oscillates many times while its centre of mass crosses the tracker, so those two figures use one resolved trajectory at each benchmark, with the initial conditions stated in the captions.
Figure~4 is reconstructed MadGraph event {ctx['ew_index']}, which has pair \(p_T = \SI{{{ctx['ew_pt']:.0f}}}{{\GeV}}\) and passes the hit-pattern cut.
Figure~5 uses every fiducial event in the two matched samples for which the plane finder returned a plane.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{fig2_angular_momentum.png}}
\caption{{Internal angular momentum \(|L|/\hbar\) and quirk separation \(d\) versus the transverse radius of the pair centre of mass.
The quirks are produced back to back in their centre of mass with speed \(v = 0.53\) along the axis \((0, 0.2, 1)\), and the pair is boosted by \(\boldsymbol{{\beta}} = (0.75, 0.05, 0.20)\).
The mass is \SI{{800}}{{\GeV}} and the string scale is \SI{{4}}{{\keV}}.
On this trajectory \(|L|/\hbar\) reaches \({sci(ew['l_max'])}\), and \(d\) runs from \SI{{{ew['d_min']:.3f}}}{{\mm}} to \SI{{{ew['d_max']:.3f}}}{{\mm}}.
This is the same pair of quantities as Figure~2 of the paper.}}
\label{{fig:l}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{fig3_plane_rotation.png}}
\caption{{Angle \(\Delta\phi\) between the instantaneous quirk plane and the plane at the start of the trajectory.
The electroweak curve uses the same initial conditions as Figure~\ref{{fig:l}} and reaches \({sci(ew['phi_max'])}\)~\si{{\rad}}.
The colored curve has mass \SI{{1800}}{{\GeV}}, speed \(v = 0.42\) along \((0.15, 1, 0.1)\), and boost \(\boldsymbol{{\beta}} = (0.70, 0.10, 0.15)\).
It reaches \({sci(qcd['phi_max'])}\)~\si{{\rad}}.
As in Figure~3 of the paper, the rotation remains small compared with the angular size of a tracker hit.}}
\label{{fig:phi}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{fig4_plane_projection.png}}
\caption{{Hits of the electroweak MadGraph event projected onto the reconstructed \((\boldsymbol{{n}}_3,\boldsymbol{{n}}_2)\) plane.
The dotted curves are the eight barrel layers.
The fit gives \(\Delta s = \SI{{{plane['delta_s_cm']:.4f}}}{{\cm}}\) and \(\Delta w = \SI{{{plane['delta_w_cm']:.4f}}}{{\cm}}\), from {plane['n_hits']} hits.
The inset shows the strip, as in Figure~4 of the paper.}}
\label{{fig:plane}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{fig5_delta_s_delta_w.png}}
\caption{{Thickness \(\Delta s\) and width \(\Delta w\) for fiducial MadGraph events in which the plane finder returned a plane, before the hit-pattern requirement.
The dashed line on the left is the signal cut \(\Delta s < \SI{{0.01}}{{\cm}}\).
The dashed line on the right is \(\Delta w < \SI{{1}}{{\cm}}\).
There are {n_ew} electroweak events and {n_qcd} colored events in the histograms.
Figure~5 of the paper also shows the pileup and photon-conversion background, which was not generated in this run, and its colored curve is at \SI{{2}}{{\keV}} rather than \SI{{4}}{{\keV}}.}}
\label{{fig:widths}}
\end{{figure}}

\end{{document}}
"""
    # The rf-string doubled braces for LaTeX. A few siunitx units need a single backslash
    # that the raw f-string already preserved. TeV was written as \TeV via {{TeV}} -> {TeV}
    # only where doubled. Check the beam-energy line: \(\sqrt{{s}}\) becomes \sqrt{s}. Good.
    (REPORT_DIR / "quirk_madgraph_report.tex").write_text(text)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {
        "ew_800": json.loads((MG_DIR / "cutflow_ew_800_lambda4000_n2000.json").read_text()),
        "qcd_1800": json.loads((MG_DIR / "cutflow_qcd_1800_lambda4000_n2000.json").read_text()),
    }
    rows = {key: load_rows(MG_DIR / f"cutflow_{key}_lambda4000_n2000_events.csv") for key in summaries}
    widths = {key: load_widths(MG_DIR / f"cutflow_{key}_lambda4000_n2000_events.csv") for key in summaries}
    figure_widths(widths)

    chosen = {key: pick_event(rows[key]) for key in summaries}
    ew_records = parse_hepmc(Path(summaries["ew_800"]["madgraph_events_file"]), 17)
    ew_index = int(chosen["ew_800"]["event"])
    ew_detector, ew_mg_trajectory = propagate_record(ew_records[ew_index], 800.0)
    plane = figure_plane(ew_detector, ew_mg_trajectory, 170802243 + ew_index * 1_000_003)

    _, ew_traj = _REPORT.simulate_pair(
        800.0, 4000.0, 0.53, np.array([0.0, 0.2, 1.0]), np.array([0.75, 0.05, 0.20])
    )
    _, qcd_traj = _REPORT.simulate_pair(
        1800.0, 4000.0, 0.42, np.array([0.15, 1.0, 0.1]), np.array([0.70, 0.10, 0.15])
    )
    ew_curves = _REPORT.trajectory_curves(ew_traj)
    qcd_curves = _REPORT.trajectory_curves(qcd_traj)
    figure_dynamics(ew_curves, qcd_curves)
    notes = {"ew_800": curve_notes(ew_curves), "qcd_1800": curve_notes(qcd_curves)}
    ctx = {
        "rows": {key: cut_rows(summary) for key, summary in summaries.items()},
        "eff": {key: efficiencies(summary) for key, summary in summaries.items()},
        "notes": {"ew": notes["ew_800"], "qcd": notes["qcd_1800"]},
        "plane": plane,
        "n_widths": {key: int(len(value[0])) for key, value in widths.items()},
        "ew_index": int(chosen["ew_800"]["event"]),
        "qcd_index": int(chosen["qcd_1800"]["event"]),
        "ew_pt": float(chosen["ew_800"]["pair_pt_gev"]),
        "qcd_pt": float(chosen["qcd_1800"]["pair_pt_gev"]),
    }
    write_tex(ctx)
    print(json.dumps({
        "chosen": {"ew": ctx["ew_index"], "qcd": ctx["qcd_index"]},
        "notes": ctx["notes"],
        "plane": plane,
        "n_widths": ctx["n_widths"],
        "eff": ctx["eff"],
    }, indent=2))
    print(f"wrote {REPORT_DIR / 'quirk_madgraph_report.tex'}")


if __name__ == "__main__":
    main()
