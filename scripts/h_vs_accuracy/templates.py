"""Analytic canonical templates for the H-vs-accuracy experiment.

Each template is a function ``y(phase)`` defined on phase in [0, 1).  The
shapes are representative rather than physically calibrated: what the
paper's Table 1 demonstrates is the *range* of Fourier-decay behaviour
across realistic fixed-shape signals (from exactly band-limited
ellipsoidal+beaming to box-like transits), not high-precision agreement
with any specific astrophysical model.

The shapes are designed so that wraparound discontinuities are absent:
each template returns to its baseline at phase = +/- 1/2, so the L^2
spectrum reflects intrinsic shape complexity rather than the
discontinuity at the period boundary.
"""

from __future__ import annotations

import numpy as np


# ---------- RR Lyrae ----------

_RRAB_TAU_RISE = 0.06
_RRAB_TAU_DECAY = 0.35


def rrab(phase):
    """Asymmetric exponential pulse: fast rise, slow decline.

    Continuous at phase = +/- 1/2 by subtracting the wraparound tail of
    each branch.  Derivative is discontinuous at phase = 0 (the peak),
    so the Fourier coefficients decay roughly as ~ 1/n^2.
    """
    phi = ((np.asarray(phase, dtype=float) + 0.5) % 1.0) - 0.5
    rise = np.exp(phi / _RRAB_TAU_RISE) - np.exp(-0.5 / _RRAB_TAU_RISE)
    decay = np.exp(-phi / _RRAB_TAU_DECAY) - np.exp(-0.5 / _RRAB_TAU_DECAY)
    y = np.where(phi < 0, rise, decay)
    peak = 1.0 - min(np.exp(-0.5 / _RRAB_TAU_RISE),
                     np.exp(-0.5 / _RRAB_TAU_DECAY))
    return y / peak


def rrc(phase):
    """Skewed sinusoid: smooth, band-limited-in-practice."""
    p = np.asarray(phase, dtype=float)
    skew = 0.15
    return -np.cos(2.0 * np.pi * (p - skew * np.sin(2.0 * np.pi * p)))


# ---------- Transit / Eclipse ----------

def _trapezoid_eclipse(phase, phi0, half_flat, taper, depth):
    """Symmetric trapezoid centred on phi0 (linear ingress/egress)."""
    phi = ((np.asarray(phase, dtype=float) - phi0 + 0.5) % 1.0) - 0.5
    abs_phi = np.abs(phi)
    y = np.zeros_like(phi)
    in_full = abs_phi < half_flat
    in_taper = (~in_full) & (abs_phi < half_flat + taper)
    y[in_full] = -depth
    y[in_taper] = -depth * (1.0 - (abs_phi[in_taper] - half_flat) / taper)
    return y


def transit(phase):
    """Schematic planet transit (trapezoidal, sharp linear ingress/egress).

    The 1/n^2 spectral decay (set by the second-derivative discontinuity
    at ingress/egress corners) is the relevant feature; actual
    limb-darkened transits have smoother edges and slightly lower H
    requirements than this schematic.
    """
    return _trapezoid_eclipse(phase, phi0=0.0,
                              half_flat=0.04, taper=0.02, depth=1.0)


def eb_detached(phase):
    """Detached eclipsing binary: trapezoidal primary + secondary."""
    primary = _trapezoid_eclipse(phase, phi0=0.0,
                                 half_flat=0.04, taper=0.02, depth=1.0)
    secondary = _trapezoid_eclipse(phase, phi0=0.5,
                                   half_flat=0.03, taper=0.02, depth=0.3)
    return primary + secondary


# ---------- Ellipsoidal + Doppler beaming ----------

def ellipsoidal_beaming(phase):
    """Beaming (1f) + ellipsoidal (2f) modulation; exactly band-limited at H=2."""
    p = np.asarray(phase, dtype=float)
    beaming = 0.3 * np.sin(2.0 * np.pi * p)
    ellipsoidal = 1.0 * np.cos(4.0 * np.pi * p)
    return beaming + ellipsoidal


# ---------- Registry ----------

TEMPLATES = {
    "rrab": rrab,
    "rrc": rrc,
    "transit": transit,
    "eb_detached": eb_detached,
    "ellipsoidal_beaming": ellipsoidal_beaming,
}

PRETTY = {
    "rrab": r"RR Lyrae ab",
    "rrc": r"RR Lyrae c",
    "transit": r"Planet transit",
    "eb_detached": r"Detached EB",
    "ellipsoidal_beaming": r"Ellipsoidal + beaming",
}


def _preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(len(TEMPLATES), 1,
                             figsize=(6, 1.6 * len(TEMPLATES)),
                             sharex=True)
    phase = np.linspace(0.0, 1.0, 4096, endpoint=False)
    for ax, (key, fn) in zip(axes, TEMPLATES.items()):
        y = fn(phase)
        ax.plot(phase, y, lw=1.4, color="C0")
        ax.set_ylabel(PRETTY[key], fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("phase")
    fig.suptitle("Canonical templates (un-normalized)", fontsize=10)
    fig.tight_layout()
    out = Path(__file__).parent / "results" / "templates_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    _preview()
