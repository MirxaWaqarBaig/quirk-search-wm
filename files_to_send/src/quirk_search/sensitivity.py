"""Cross sections and the 3-event / 5-event contours of Fig. 6.

Drell-Yan is leading-order s-channel photon production of a colour-singlet
Dirac fermion. QCD production is leading-order gg and qqbar production of a
colour-triplet Dirac fermion. Parton densities are a toy set, and the QCD
rate is scaled so that a 173 GeV quark pair has the measured 13 TeV ttbar
cross section of 831 pb. These are not the MadGraph samples used in the paper.
The HSCP and monojet curves on Fig. 6 came from a private communication and
are not reproduced.
"""

from __future__ import annotations

import math

import numpy as np

ALPHA_EM = 1.0 / 128.0
GEV2_TO_PB = 3.893793656e8
SQRT_S_GEV = 13000.0
TOP_MASS_GEV = 173.0
TOP_CROSS_SECTION_PB = 831.0

_PDF_CACHE: dict | None = None
_QCD_SCALE: float | None = None


def alpha_s(scale_gev: float) -> float:
    """One-loop alpha_s with alpha_s(m_Z) = 0.118 and five flavours."""
    mz = 91.1876
    alpha_z = 0.118
    nf = 5.0
    beta0 = (11.0 * 3.0 - 2.0 * nf) / 3.0
    scale = max(float(scale_gev), 1.0)
    return 1.0 / (1.0 / alpha_z + (beta0 / (4.0 * math.pi)) * math.log((scale * scale) / (mz * mz)))


def _trap(samples: np.ndarray, values: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(values, samples))


def _pdfs():
    """Valence normalised to 2 and 1. Sea and gluon carry the rest of the momentum."""
    global _PDF_CACHE
    if _PDF_CACHE is not None:
        return _PDF_CACHE
    x = np.exp(np.linspace(math.log(1e-4), math.log(0.999), 400))

    def integrate(raw):
        return _trap(x, raw(x))

    uv_raw = lambda z: z**0.5 * (1.0 - z) ** 3.5
    dv_raw = lambda z: z**0.5 * (1.0 - z) ** 4.5
    sea_raw = lambda z: z ** (-0.2) * (1.0 - z) ** 7.0
    g_raw = lambda z: z ** (-0.3) * (1.0 - z) ** 5.0
    uv = 2.0 / integrate(uv_raw)
    dv = 1.0 / integrate(dv_raw)
    # Leave about half the momentum for gluons after valence and a modest sea.
    sea_norm = 0.15 / integrate(lambda z: z * sea_raw(z))
    valence_momentum = integrate(lambda z: z * (uv * uv_raw(z) + dv * dv_raw(z)))
    sea_momentum = 5.0 * integrate(lambda z: z * sea_norm * sea_raw(z))
    gluon_momentum = max(0.05, 1.0 - valence_momentum - sea_momentum)
    g_norm = gluon_momentum / integrate(lambda z: z * g_raw(z))
    _PDF_CACHE = {
        "uv": uv,
        "dv": dv,
        "sea": sea_norm,
        "g": g_norm,
        "uv_raw": uv_raw,
        "dv_raw": dv_raw,
        "sea_raw": sea_raw,
        "g_raw": g_raw,
    }
    return _PDF_CACHE


def pdf_at(x: np.ndarray) -> dict[str, np.ndarray]:
    tables = _pdfs()
    x = np.asarray(x, dtype=float)
    sea = tables["sea"] * tables["sea_raw"](x)
    return {
        "u": tables["uv"] * tables["uv_raw"](x) + sea,
        "d": tables["dv"] * tables["dv_raw"](x) + sea,
        "s": sea,
        "c": sea,
        "b": sea,
        "ubar": sea,
        "dbar": sea,
        "sbar": sea,
        "cbar": sea,
        "bbar": sea,
        "g": tables["g"] * tables["g_raw"](x),
    }


_QUARK_CHARGE2 = {"u": 4.0 / 9.0, "d": 1.0 / 9.0, "s": 1.0 / 9.0, "c": 4.0 / 9.0, "b": 1.0 / 9.0}
_QQBAR_PAIRS = (("u", "ubar"), ("d", "dbar"), ("s", "sbar"), ("c", "cbar"), ("b", "bbar"))


def sigma_hat_dy_gev2(shat: float, mass_gev: float, quark_charge2: float, quirk_charge: float = 1.0) -> float:
    """Colour-averaged qqbar -> quirk quirkbar via a virtual photon."""
    if shat <= 4.0 * mass_gev * mass_gev:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 * mass_gev * mass_gev / shat)
    prefactor = 4.0 * math.pi * ALPHA_EM * ALPHA_EM / (9.0 * shat)
    phase = beta * (3.0 - beta * beta) / 2.0
    return prefactor * quark_charge2 * quirk_charge * quirk_charge * phase


def sigma_hat_qq_qcd_gev2(shat: float, mass_gev: float, coupling: float) -> float:
    if shat <= 4.0 * mass_gev * mass_gev:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 * mass_gev * mass_gev / shat)
    return (4.0 * math.pi * coupling * coupling) / (27.0 * shat) * beta * (3.0 - beta * beta)


def sigma_hat_gg_qcd_gev2(shat: float, mass_gev: float, coupling: float) -> float:
    """LO gg -> Q Qbar. The bracket is the standard Dawson-Ellis-Nason form."""
    if shat <= 4.0 * mass_gev * mass_gev:
        return 0.0
    rho = 4.0 * mass_gev * mass_gev / shat
    beta = math.sqrt(max(0.0, 1.0 - rho))
    beta = min(beta, 0.999999)
    log_term = math.log((1.0 + beta) / (1.0 - beta))
    bracket = (1.0 + rho + 0.25 * rho * rho) * log_term - beta * (1.75 + 1.9375 * rho)
    if bracket < 0.0:
        return 0.0
    return (math.pi * coupling * coupling) / (3.0 * shat) * bracket


def _hadronic_grid(mass_gev: float):
    s_had = SQRT_S_GEV * SQRT_S_GEV
    threshold = 4.0 * mass_gev * mass_gev / s_had
    if threshold >= 1.0:
        return None
    n = 48
    log_lo = math.log(max(threshold, 1e-4))
    log_hi = math.log(0.999)
    xs = np.exp(np.linspace(log_lo, log_hi, n))
    dln = (log_hi - log_lo) / (n - 1)
    pdf = pdf_at(xs)
    return s_had, xs, dln, pdf


def _partonic_weights(xs: np.ndarray, dln: float, s_had: float, mass_gev: float):
    x1 = xs[:, None]
    x2 = xs[None, :]
    shat = x1 * x2 * s_had
    mask = shat > 4.0 * mass_gev * mass_gev
    weight = x1 * x2 * dln * dln
    return shat, mask, weight


def dy_cross_section_pb(mass_gev: float, quirk_charge: float = 1.0) -> float:
    grid = _hadronic_grid(mass_gev)
    if grid is None:
        return 0.0
    s_had, xs, dln, pdf = grid
    shat, mask, weight = _partonic_weights(xs, dln, s_had, mass_gev)
    beta = np.zeros_like(shat)
    beta[mask] = np.sqrt(1.0 - 4.0 * mass_gev * mass_gev / shat[mask])
    phase = beta * (3.0 - beta * beta) / 2.0
    prefactor = 4.0 * np.pi * ALPHA_EM * ALPHA_EM / (9.0 * np.maximum(shat, 1e-30))
    total = 0.0
    charge = quirk_charge * quirk_charge
    for name, anti in _QQBAR_PAIRS:
        luminosity = pdf[name][:, None] * pdf[anti][None, :]
        partonic = prefactor * _QUARK_CHARGE2[name] * charge * phase
        total += float(np.sum(weight * luminosity * partonic * mask))
    return total * GEV2_TO_PB


def _qcd_unscaled_pb(mass_gev: float) -> float:
    grid = _hadronic_grid(mass_gev)
    if grid is None:
        return 0.0
    s_had, xs, dln, pdf = grid
    shat, mask, weight = _partonic_weights(xs, dln, s_had, mass_gev)
    coupling = alpha_s(max(mass_gev, 1.0))
    beta = np.zeros_like(shat)
    beta[mask] = np.sqrt(1.0 - 4.0 * mass_gev * mass_gev / shat[mask])
    beta = np.minimum(beta, 0.999999)
    rho = np.zeros_like(shat)
    rho[mask] = 4.0 * mass_gev * mass_gev / shat[mask]
    qq = np.zeros_like(shat)
    qq[mask] = (4.0 * np.pi * coupling * coupling) / (27.0 * shat[mask]) * beta[mask] * (3.0 - beta[mask] ** 2)
    log_term = np.zeros_like(shat)
    safe = mask & (beta < 0.999999)
    log_term[safe] = np.log((1.0 + beta[safe]) / (1.0 - beta[safe]))
    bracket = (1.0 + rho + 0.25 * rho * rho) * log_term - beta * (1.75 + 1.9375 * rho)
    gg = np.zeros_like(shat)
    positive = mask & (bracket > 0.0)
    gg[positive] = (np.pi * coupling * coupling) / (3.0 * shat[positive]) * bracket[positive]
    total = float(np.sum(weight * (pdf["g"][:, None] * pdf["g"][None, :]) * gg))
    for name, anti in _QQBAR_PAIRS:
        luminosity = pdf[name][:, None] * pdf[anti][None, :]
        total += float(np.sum(weight * luminosity * qq))
    return total * GEV2_TO_PB


def qcd_scale_factor() -> float:
    """Scale the toy LO QCD rate onto sigma(ttbar, 13 TeV) = 831 pb."""
    global _QCD_SCALE
    if _QCD_SCALE is None:
        raw = _qcd_unscaled_pb(TOP_MASS_GEV)
        if raw <= 0.0:
            raise RuntimeError("QCD cross section vanished at the top mass")
        _QCD_SCALE = TOP_CROSS_SECTION_PB / raw
    return _QCD_SCALE


def qcd_cross_section_pb(mass_gev: float) -> float:
    return _qcd_unscaled_pb(mass_gev) * qcd_scale_factor()


def cross_section_pb(mass_gev: float, mode: str) -> float:
    if mode == "dy":
        return dy_cross_section_pb(mass_gev)
    if mode == "qcd":
        return qcd_cross_section_pb(mass_gev)
    raise ValueError(f"unknown production mode {mode}")


def expected_events(
    sigma_pb: float,
    luminosity_fb: float,
    epsilon: float,
    charged_fraction: float = 1.0,
) -> float:
    """pb times fb^-1, converted by 1000 pb per fb."""
    return float(sigma_pb) * float(luminosity_fb) * 1000.0 * float(epsilon) * float(charged_fraction)


def contour_label(n_events: float, discovery: float = 5.0, exclusion: float = 3.0) -> str:
    if n_events >= discovery:
        return "discovery"
    if n_events >= exclusion:
        return "exclusion"
    return "below"
