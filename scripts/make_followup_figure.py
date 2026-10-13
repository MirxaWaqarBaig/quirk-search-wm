#!/usr/bin/env python3
"""Figure for the trajectory-clarification note."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quirk_search.detector import default_detector
from quirk_search.dynamics import HBARC_GEV_MM, initial_pair_state, propagate_pair

OUT = ROOT / "results" / "report" / "followup_figures"


def curves(b_field: float):
    detector = default_detector()
    state = initial_pair_state(
        800.0,
        4000.0,
        0.53,
        np.array([0.0, 0.2, 1.0]),
        np.array([0.75, 0.05, 0.20]),
    )
    traj = propagate_pair(
        state,
        b_field_t=b_field,
        t_max_ns=detector.timing_ns,
        outer_radius_mm=detector.outer_radius_mm,
    )
    x1 = traj.quirk.xyz
    x2 = traj.antiquirk.xyz
    p1 = traj.p_quirk
    p2 = traj.p_antiquirk
    xcm = 0.5 * (x1 + x2)
    step = max(1, len(x1) // 4000)
    sl = slice(None, None, step)
    separation = np.linalg.norm(x1 - x2, axis=1)
    angular = np.cross(x1 - xcm, p1) + np.cross(x2 - xcm, p2)
    return {
        "r": np.hypot(xcm[sl, 0], xcm[sl, 1]),
        "d": separation[sl],
        "L": np.linalg.norm(angular, axis=1)[sl] / HBARC_GEV_MM,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    on = curves(2.0)
    off = curves(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    axes[0].plot(on["r"], on["L"], color="#1f4e79", lw=1.0, label="$B = 2$ T")
    axes[0].plot(off["r"], off["L"], color="#c47b2b", lw=1.0, label="$B = 0$")
    axes[0].set_xlabel("Transverse radius of the centre of mass [mm]")
    axes[0].set_ylabel(r"Laboratory $|L|/\hbar$ about the midpoint")
    axes[0].set_title("Same event, with and without the field")
    axes[0].legend(frameon=False)
    axes[1].plot(on["r"], on["d"], color="#1f4e79", lw=1.0, label="$B = 2$ T")
    axes[1].plot(off["r"], off["d"], color="#c47b2b", lw=1.0, label="$B = 0$")
    axes[1].axhline(3.537, color="#333333", ls="--", lw=1.0, label="CM amplitude, 3.54 mm")
    axes[1].set_xlabel("Transverse radius of the centre of mass [mm]")
    axes[1].set_ylabel("Laboratory separation $d$ [mm]")
    axes[1].set_title("Separation at equal laboratory times")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "lab_angular_momentum.png", dpi=160)
    plt.close(fig)
    print(OUT / "lab_angular_momentum.png")


if __name__ == "__main__":
    main()
