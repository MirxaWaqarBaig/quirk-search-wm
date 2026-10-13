#!/usr/bin/env python3
"""Build the cut-flow report, Wilson intervals, and Figures 2-5."""

from __future__ import annotations

import csv
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
from quirk_search.dynamics import HBARC_GEV_MM, initial_pair_state, propagate_pair
from quirk_search.plane_finder import PaperCuts, find_plane

BENCHMARK_DIR = ROOT / "results" / "benchmarks"
FIGURE_DIR = ROOT / "results" / "figures"
REPORT_DIR = ROOT / "results" / "report"
Z95 = 1.959963984540054

PAPER_TABLE1 = {
    "dy_800": {
        "label": "Electroweak, 800 GeV",
        "lambda_note": "4 keV, the Table I row",
        "epsilon_trigger": 0.10,
        "epsilon_fiducial": 0.28,
        "epsilon_reco": 0.72,
    },
    "qcd_1800": {
        "label": "Colored, 1800 GeV",
        "lambda_note": "no 4 keV row; 0.59 at 3 keV and 0.74 at 5 keV",
        "epsilon_trigger": 0.24,
        "epsilon_fiducial": 0.28,
        "epsilon_reco": None,
    },
}


def wilson(k: int, n: int) -> tuple[float, float, float, float]:
    """Return estimate, standard error, and Wilson 95% lower and upper bounds."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p = k / n
    se = math.sqrt(p * (1.0 - p) / n)
    z2 = Z95 * Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return p, se, max(0.0, center - half), min(1.0, center + half)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def load_signal_widths(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ds, dw = [], []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["trigger"] != "1" or row["fiducial"] != "1":
                continue
            if row["delta_s_mm"] == "" or row["delta_w_mm"] == "":
                continue
            ds.append(float(row["delta_s_mm"]) / 10.0)  # mm -> cm
            dw.append(float(row["delta_w_mm"]) / 10.0)
    return np.asarray(ds), np.asarray(dw)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.3f}"


def fmt_ci(low: float, high: float) -> str:
    if math.isnan(low):
        return "—"
    return f"[{low:.3f}, {high:.3f}]"


def cut_rows(summary: dict) -> list[dict]:
    cuts = summary["cutflow"]["cuts"]
    rows = []
    previous_n = None
    for cut in cuts:
        k = int(cut["n_pass"])
        n = int(previous_n) if previous_n is not None else k
        if cut["cut"] == "generated":
            estimate, se, low, high = 1.0, 0.0, 1.0, 1.0
            n = k
        else:
            estimate, se, low, high = wilson(k, n)
        rows.append(
            {
                "cut": cut["cut"],
                "definition": cut["definition"],
                "n_pass": k,
                "n_previous": n,
                "estimate": estimate,
                "se": se,
                "low": low,
                "high": high,
            }
        )
        previous_n = k
    return rows


def efficiency_block(summary: dict) -> dict:
    n_gen = int(summary["n_generated"])
    n_trig = int(summary["n_triggered"])
    n_fid = int(summary["n_fiducial"])
    n_reco = int(summary["n_reconstructed"])
    keys = {
        "epsilon_trigger": (n_trig, n_gen),
        "epsilon_fiducial": (n_fid, n_trig),
        "epsilon_reco": (n_reco, n_fid),
        "epsilon_total": (n_reco, n_gen),
    }
    out = {}
    for name, (k, n) in keys.items():
        estimate, se, low, high = wilson(k, n)
        out[name] = {"k": k, "n": n, "estimate": estimate, "se": se, "low": low, "high": high}
    return out


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def simulate_pair(mass_gev: float, lambda_ev: float, v_internal: float, axis: np.ndarray, beta: np.ndarray):
    detector = default_detector()
    detector.beamspot_sigma_z_mm = 0.0
    state = initial_pair_state(
        mass_gev=mass_gev,
        lambda_ev=lambda_ev,
        v_internal=v_internal,
        axis=axis,
        beta_lab=beta,
        vertex_mm=np.zeros(3),
    )
    trajectory = propagate_pair(
        state,
        b_field_t=detector.b_field_t,
        t_max_ns=detector.timing_ns,
        outer_radius_mm=detector.outer_radius_mm,
    )
    return detector, trajectory


def trajectory_curves(trajectory) -> dict[str, np.ndarray]:
    x1 = trajectory.quirk.xyz
    x2 = trajectory.antiquirk.xyz
    p1 = trajectory.p_quirk
    p2 = trajectory.p_antiquirk
    xcm = 0.5 * (x1 + x2)
    r_cm = np.hypot(xcm[:, 0], xcm[:, 1])
    separation = np.linalg.norm(x1 - x2, axis=1)
    angular = np.cross(x1 - xcm, p1) + np.cross(x2 - xcm, p2)
    l_over_hbar = np.linalg.norm(angular, axis=1) / HBARC_GEV_MM
    relative_x = x1 - x2
    relative_p = p1 - p2
    normal = np.cross(relative_x, relative_p)
    norm = np.linalg.norm(normal, axis=1)
    valid = (norm > 1e-8) & (separation > 0.05)
    delta_phi = np.full(len(r_cm), np.nan)
    if np.any(valid):
        reference = normal[int(np.argmax(valid))]
        reference = reference / np.linalg.norm(reference)
        hat = np.divide(normal, norm[:, None], out=np.zeros_like(normal), where=norm[:, None] > 1e-8)
        cosine = np.clip(np.abs(hat @ reference), 0.0, 1.0)
        delta_phi[valid] = np.arccos(cosine[valid])
    return {"r_cm_mm": r_cm, "separation_mm": separation, "l_over_hbar": l_over_hbar, "delta_phi": delta_phi}


def project_layers(normal_long: np.ndarray, normal_width: np.ndarray, radii: list[float]) -> list[np.ndarray]:
    curves = []
    phi = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    for radius in radii:
        circle = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)])
        curves.append(np.column_stack([circle @ normal_long, circle @ normal_width]))
    return curves


def figure_dynamics(curves: dict[str, dict[str, np.ndarray]]) -> None:
    ew = curves["ew"]
    qcd = curves["qcd"]
    fig, ax_l = plt.subplots(figsize=(7.2, 4.6))
    ax_d = ax_l.twinx()
    ax_l.plot(ew["r_cm_mm"], ew["l_over_hbar"], color="#1f4e79", lw=1.2, label=r"$|L|/\hbar$")
    ax_d.plot(ew["r_cm_mm"], ew["separation_mm"], color="#c47b2b", lw=1.2, label="separation $d$")
    ax_l.set_xlabel("Transverse radius of the quirk-pair centre of mass [mm]")
    ax_l.set_ylabel(r"Internal angular momentum $|L|/\hbar$")
    ax_d.set_ylabel("Quirk separation $d$ [mm]")
    ax_l.set_title("Angular momentum and separation, electroweak benchmark")
    lines = ax_l.get_lines() + ax_d.get_lines()
    ax_l.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_angular_momentum.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ew["r_cm_mm"], ew["delta_phi"], color="#1f4e79", lw=1.2, label="800 GeV electroweak, 4 keV")
    ax.plot(qcd["r_cm_mm"], qcd["delta_phi"], color="#b85c38", lw=1.2, label="1800 GeV colored, 4 keV")
    ax.set_xlabel("Transverse radius of the quirk-pair centre of mass [mm]")
    ax.set_ylabel(r"Plane rotation $\Delta\phi$ [rad]")
    ax.set_title("Rotation of the quirk plane in a 2 T field")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_plane_rotation.png", dpi=160)
    plt.close(fig)


def figure_plane(detector, trajectory) -> dict:
    measured = measure_pair(detector, trajectory, np.random.default_rng(4))
    result = find_plane(measured.reconstructed, PaperCuts())
    if not result.found:
        raise RuntimeError("representative event did not yield a plane for Figure 4")
    points = np.array([[hit.x, hit.y, hit.z] for hit in measured.reconstructed], dtype=float)
    selected = points[result.indices]
    long_coord = selected @ result.n3
    width_coord = selected @ result.n2
    radii = [layer.radius_mm for layer in detector.layers]
    ellipses = project_layers(result.n3, result.n2, radii)

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
    inset.set_xlim(np.min(long_coord) - 5.0, np.max(long_coord) + 5.0)
    pad = max(1.0, 3.0 * result.delta_w_mm)
    inset.set_ylim(np.min(width_coord) - pad, np.max(width_coord) + pad)
    inset.set_title(rf"$\Delta w = {result.delta_w_mm / 10.0:.3f}$ cm", fontsize=8)
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_plane_projection.png", dpi=160)
    plt.close(fig)
    return {"delta_s_cm": result.delta_s_mm / 10.0, "delta_w_cm": result.delta_w_mm / 10.0, "n_hits": int(len(result.indices))}


def figure_widths(samples: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    colors = {"dy_800": "#1f4e79", "qcd_1800": "#b85c38"}
    labels = {"dy_800": "Electroweak, 800 GeV, 4 keV", "qcd_1800": "Colored, 1800 GeV, 4 keV"}
    for key, (ds, dw) in samples.items():
        axes[0].hist(ds, bins=40, range=(0.0, 0.02), histtype="step", lw=1.6, color=colors[key], label=f"{labels[key]} (N={len(ds)})")
        axes[1].hist(dw, bins=40, range=(0.0, 1.2), histtype="step", lw=1.6, color=colors[key], label=labels[key])
    axes[0].axvline(0.01, color="#333333", ls="--", lw=1.0, label=r"signal cut $\Delta s < 0.01$ cm")
    axes[1].axvline(1.0, color="#333333", ls="--", lw=1.0, label=r"width cut $\Delta w < 1$ cm")
    axes[0].set_xlabel(r"Plane thickness $\Delta s$ [cm]")
    axes[1].set_xlabel(r"Oscillation width $\Delta w$ [cm]")
    axes[0].set_ylabel("Fiducial events with a reconstructed plane")
    axes[1].set_ylabel("Fiducial events with a reconstructed plane")
    axes[0].set_title(r"$\Delta s$")
    axes[1].set_title(r"$\Delta w$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig5_delta_s_delta_w.png", dpi=160)
    plt.close(fig)


def comparison_sentence(name: str, paper: float | None, ours: dict) -> str:
    if paper is None:
        return "The paper does not tabulate this entry at 4 keV."
    inside = ours["low"] <= paper <= ours["high"]
    place = "inside" if inside else "outside"
    return f"The Table I value {paper:.2f} lies {place} this interval."


def write_report(summaries: dict[str, dict], efficiencies: dict[str, dict], plane_info: dict, curve_notes: dict) -> None:
    dy = summaries["dy_800"]
    qcd = summaries["qcd_1800"]
    lines = []
    lines.append("# Quirk plane-search report")
    lines.append("")
    lines.append("Comparison of the 2000-event electroweak and colored samples with Knapen, Lou, Papucci, and Setford, arXiv:1708.02243, Table I, together with Wilson 95% intervals and the analogues of Figures 2–5.")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    lines.append("Both samples use the straight-string propagator, a 2 T solenoid, and the eight-layer barrel. Each sample has 2000 generated events. The electroweak sample has mass 800 GeV. The colored sample has mass 1800 GeV. Both use a string scale of 4 keV. Momenta come from the built-in sampler. MadGraph and Pythia were not used for these events.")
    lines.append("")
    lines.append("## How the efficiencies are defined")
    lines.append("")
    lines.append("An event is counted in a cut only if it passed every earlier cut. The three efficiencies in the paper are ratios of those counts.")
    lines.append("")
    lines.append(markdown_table(
        ["Efficiency", "Definition", "Formula"],
        [
            ["Trigger", "Pair transverse momentum above 200 GeV.", r"\(\varepsilon_{\mathrm{trig}} = N_{\mathrm{trig}} / N_{\mathrm{gen}}\)"],
            ["Fiducial", "Both quirks hit all eight barrel layers, with every hit time under 25 ns.", r"\(\varepsilon_{\mathrm{fid}} = N_{\mathrm{fid}} / N_{\mathrm{trig}}\)"],
            ["Reconstruction", "The plane finder and the final hit pattern both succeed.", r"\(\varepsilon_{\mathrm{reco}} = N_{\mathrm{reco}} / N_{\mathrm{fid}}\)"],
            ["Total", "Product of the three factors. It is also the fraction of all generated events that pass the last cut.", r"\(\varepsilon = \varepsilon_{\mathrm{trig}}\varepsilon_{\mathrm{fid}}\varepsilon_{\mathrm{reco}}\)"],
        ],
    ))
    lines.append("")
    lines.append("The reconstruction count is the number of events that pass the hit-pattern cut. Because the flow is sequential, that count already requires the trigger, the fiducial requirement, the seed windows, the factor-of-three growth test, \(\\Delta w < 1\\,\\mathrm{cm}\), and \(\\Delta s < 0.01\\,\\mathrm{cm}\).")
    lines.append("")
    lines.append("## Wilson 95% intervals")
    lines.append("")
    lines.append("Each efficiency is a binomial proportion \(k/n\). The quoted uncertainty is the standard error \(\\sqrt{p(1-p)/n}\). The 95% interval is the Wilson score interval,")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"\left(\hat p + \frac{z^2}{2n}\right)\frac{1}{1+z^2/n}")
    lines.append(r"\;\pm\;")
    lines.append(r"\frac{z}{1+z^2/n}")
    lines.append(r"\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}},")
    lines.append(r"\]")
    lines.append("")
    lines.append("with \(z = 1.960\). For a conditional cut, \(k\) is the number passing that cut and \(n\) is the number that reached it.")
    lines.append("")

    for key, title in (("dy_800", "Electroweak sample, 800 GeV"), ("qcd_1800", "Colored sample, 1800 GeV")):
        lines.append(f"### {title}")
        lines.append("")
        rows = []
        for item in cut_rows(summaries[key]):
            rows.append([
                item["cut"],
                str(item["n_pass"]),
                str(item["n_previous"]),
                fmt(item["estimate"]),
                fmt(item["se"]),
                fmt_ci(item["low"], item["high"]),
            ])
        lines.append(markdown_table(
            ["Cut", "N pass", "N reaching the cut", "Efficiency", "Standard error", "Wilson 95% CI"],
            rows,
        ))
        lines.append("")
        eff = efficiencies[key]
        eff_rows = []
        for name, label in (
            ("epsilon_trigger", "Trigger"),
            ("epsilon_fiducial", "Fiducial"),
            ("epsilon_reco", "Reconstruction"),
            ("epsilon_total", "Total"),
        ):
            item = eff[name]
            eff_rows.append([
                label,
                f"{item['k']} / {item['n']}",
                fmt(item["estimate"]),
                fmt(item["se"]),
                fmt_ci(item["low"], item["high"]),
            ])
        lines.append(markdown_table(
            ["Efficiency", "Count", "Estimate", "Standard error", "Wilson 95% CI"],
            eff_rows,
        ))
        lines.append("")

    lines.append("## Comparison with Table I")
    lines.append("")
    lines.append("Table I lists the electroweak efficiencies at 800 GeV and the colored efficiencies at 1800 GeV. Trigger and fiducial efficiencies in that table do not depend on the string scale. The reconstruction efficiency does. This run is at 4 keV for both samples. Table I gives the electroweak reconstruction efficiency at 4 keV. It does not give a colored row at 4 keV; the neighboring colored entries are 0.59 at 3 keV and 0.74 at 5 keV.")
    lines.append("")
    comp_rows = []
    for key in ("dy_800", "qcd_1800"):
        paper = PAPER_TABLE1[key]
        eff = efficiencies[key]
        paper_reco = "0.72" if paper["epsilon_reco"] is not None else "0.59 at 3 keV; 0.74 at 5 keV"
        comp_rows.append([
            paper["label"],
            "Trigger",
            f"{paper['epsilon_trigger']:.2f}",
            fmt(eff["epsilon_trigger"]["estimate"]),
            fmt(eff["epsilon_trigger"]["se"]),
            fmt_ci(eff["epsilon_trigger"]["low"], eff["epsilon_trigger"]["high"]),
            comparison_sentence("trigger", paper["epsilon_trigger"], eff["epsilon_trigger"]),
        ])
        comp_rows.append([
            paper["label"],
            "Fiducial",
            f"{paper['epsilon_fiducial']:.2f}",
            fmt(eff["epsilon_fiducial"]["estimate"]),
            fmt(eff["epsilon_fiducial"]["se"]),
            fmt_ci(eff["epsilon_fiducial"]["low"], eff["epsilon_fiducial"]["high"]),
            comparison_sentence("fiducial", paper["epsilon_fiducial"], eff["epsilon_fiducial"]),
        ])
        comp_rows.append([
            paper["label"],
            "Reconstruction",
            paper_reco,
            fmt(eff["epsilon_reco"]["estimate"]),
            fmt(eff["epsilon_reco"]["se"]),
            fmt_ci(eff["epsilon_reco"]["low"], eff["epsilon_reco"]["high"]),
            comparison_sentence("reco", paper["epsilon_reco"], eff["epsilon_reco"]),
        ])
    lines.append(markdown_table(
        ["Sample", "Efficiency", "Table I", "This work", "Standard error", "Wilson 95% CI", "Table I versus this interval"],
        comp_rows,
    ))
    lines.append("")
    lines.append(f"Electroweak counts are {dy['n_triggered']} triggered, {dy['n_fiducial']} fiducial, and {dy['n_reconstructed']} reconstructed, out of {dy['n_generated']}. Colored counts are {qcd['n_triggered']} triggered, {qcd['n_fiducial']} fiducial, and {qcd['n_reconstructed']} reconstructed, out of {qcd['n_generated']}.")
    lines.append("")
    lines.append("The trigger and fiducial values sit far from Table I. Those two efficiencies are properties of the momentum distribution. This run uses the built-in sampler, while Table I uses MadGraph and Pythia, so the difference is expected. The reconstruction efficiency uses the same plane cuts. For the electroweak sample the Table I value 0.72 can be compared directly with the Wilson interval above.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("### Figure 2")
    lines.append("")
    lines.append("![Angular momentum and separation](../figures/fig2_angular_momentum.png)")
    lines.append("")
    lines.append(f"Internal angular momentum and quirk separation for one electroweak event at 800 GeV and 4 keV in a 2 T field. The maximum \(|L|/\\hbar\) on the tracked trajectory is {curve_notes['ew_l_max']:.3e}, and the separation stays between {curve_notes['ew_d_min']:.3f} mm and {curve_notes['ew_d_max']:.3f} mm. This is the same pair of quantities as Figure 2 of the paper, for this benchmark rather than for their unpublished representative event.")
    lines.append("")
    lines.append("### Figure 3")
    lines.append("")
    lines.append("![Plane rotation](../figures/fig3_plane_rotation.png)")
    lines.append("")
    lines.append(f"Angle between the instantaneous quirk plane and the plane at the start of the trajectory. The electroweak curve reaches {curve_notes['ew_phi_max']:.3e} rad. The colored curve, at 1800 GeV and 4 keV, reaches {curve_notes['qcd_phi_max']:.3e} rad. The paper's Figure 3 shows this rotation for two benchmark events and finds it small compared with the tracker resolution.")
    lines.append("")
    lines.append("### Figure 4")
    lines.append("")
    lines.append("![Plane projection](../figures/fig4_plane_projection.png)")
    lines.append("")
    lines.append(f"Hits of one electroweak event projected onto the reconstructed plane, with the eight barrel layers drawn as dotted curves. The fitted thickness is \(\\Delta s = {plane_info['delta_s_cm']:.4f}\\,\\mathrm{{cm}}\) and the fitted width is \(\\Delta w = {plane_info['delta_w_cm']:.4f}\\,\\mathrm{{cm}}\), from {plane_info['n_hits']} hits. The inset is the strip, as in Figure 4 of the paper.")
    lines.append("")
    lines.append("### Figure 5")
    lines.append("")
    lines.append("![Delta s and delta w](../figures/fig5_delta_s_delta_w.png)")
    lines.append("")
    lines.append("Thickness and width for fiducial events in which the plane finder returned a plane, before requiring the final hit pattern. The dashed line on the left is the signal cut \(\\Delta s < 0.01\\,\\mathrm{cm}\). The dashed line on the right is \(\\Delta w < 1\\,\\mathrm{cm}\). The paper's Figure 5 also shows the pileup and photon-conversion background. Those background samples were not part of this 2000-event generation, so this figure shows the two signal samples only. The paper's colored curve is at 2 keV; the colored curve here is the 4 keV sample that was generated.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("The input summaries are `cutflow_dy_800_lambda4000_n2000.json`, `cutflow_dy_800_lambda4000_n2000_events.csv`, `cutflow_qcd_1800_lambda4000_n2000.json`, and `cutflow_qcd_1800_lambda4000_n2000_events.csv`.")
    lines.append("")
    report = "\n".join(lines) + "\n"
    (REPORT_DIR / "quirk_cutflow_report.md").write_text(report)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {
        "dy_800": load_summary(BENCHMARK_DIR / "cutflow_dy_800_lambda4000_n2000.json"),
        "qcd_1800": load_summary(BENCHMARK_DIR / "cutflow_qcd_1800_lambda4000_n2000.json"),
    }
    efficiencies = {key: efficiency_block(summary) for key, summary in summaries.items()}
    widths = {
        "dy_800": load_signal_widths(BENCHMARK_DIR / "cutflow_dy_800_lambda4000_n2000_events.csv"),
        "qcd_1800": load_signal_widths(BENCHMARK_DIR / "cutflow_qcd_1800_lambda4000_n2000_events.csv"),
    }
    figure_widths(widths)

    detector, ew_traj = simulate_pair(800.0, 4000.0, 0.53, np.array([0.0, 0.2, 1.0]), np.array([0.75, 0.05, 0.20]))
    _, qcd_traj = simulate_pair(1800.0, 4000.0, 0.42, np.array([0.15, 1.0, 0.1]), np.array([0.70, 0.10, 0.15]))
    ew_curves = trajectory_curves(ew_traj)
    qcd_curves = trajectory_curves(qcd_traj)
    figure_dynamics({"ew": ew_curves, "qcd": qcd_curves})
    plane_info = figure_plane(detector, ew_traj)
    finite_d = ew_curves["separation_mm"][np.isfinite(ew_curves["separation_mm"])]
    curve_notes = {
        "ew_l_max": float(np.nanmax(ew_curves["l_over_hbar"])),
        "ew_d_min": float(np.min(finite_d)),
        "ew_d_max": float(np.max(finite_d)),
        "ew_phi_max": float(np.nanmax(ew_curves["delta_phi"])),
        "qcd_phi_max": float(np.nanmax(qcd_curves["delta_phi"])),
    }
    write_report(summaries, efficiencies, plane_info, curve_notes)
    print(json.dumps({"plane": plane_info, "curves": curve_notes, "n_ds": {k: int(len(v[0])) for k, v in widths.items()}}, indent=2))
    print(f"wrote {REPORT_DIR / 'quirk_cutflow_report.md'}")


if __name__ == "__main__":
    main()
