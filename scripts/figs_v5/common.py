"""Shared helpers for figs_v5/ figures.

These figures are part of a fresh rewrite (paper_v5) of Figures 1-3 in the FTP
paper. See README.md in this directory for an overview.

NOTHING in this module is allowed to import from ``scripts/`` outside of
``scripts/figs_v5``. The only legacy artifact we reuse is the EB-template
pickle ``saved_results/eb_template_HAT-059-0780895.pkl``.
"""
from __future__ import annotations

import os
import pickle
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

# A muted "ggplot-like" sans palette. Tuned to read well on white in print.
COL_FTP = "#222222"        # near-black (FTP / lightcurve / template fit)
COL_MHLS = "#3268a8"       # muted blue (multi-harmonic Lomb-Scargle)
COL_BLS = "#8a4fbf"        # muted purple (Box Least Squares)
COL_DIAG = "#888888"       # diagonal guide in scatter plots
COL_TRUEF = "#000000"      # true-frequency dotted guide
COL_DATA = "#a82a2a"       # lightcurve data points

# Linewidths and marker sizes (tuned to look right on a single-column figure)
LW_THIN = 0.7
LW_MED = 1.0
MS_PTS = 2.0


def setup_mpl() -> None:
    """Set matplotlib rcParams shared across the v5 figures.

    Key fix (L.6, Lintott): make sure minus signs in axis tick labels render
    as proper characters rather than missing-glyph boxes. We force a font
    family that ships a real minus, and turn off the matplotlib unicode-minus
    behavior so tick labels use ASCII '-' instead of U+2212. We also ensure
    mathtext uses the regular sans font.
    """
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans", "Helvetica", "Arial", "Liberation Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,        # embed actual font glyphs in PDF
        "ps.fonttype": 42,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
    })


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
EB_TEMPLATE_PKL = os.path.join(
    PAPER_ROOT, "scripts", "saved_results", "eb_template_HAT-059-0780895.pkl"
)
PLOTS_DIR = os.path.join(PAPER_ROOT, "plots")


# ----------------------------------------------------------------------------
# Template loading
# ----------------------------------------------------------------------------

def load_eb_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """Load (c_n, s_n) from the EB-template pickle.

    The pickle is a Python-2 ASCII pickle of a 2-tuple of length-10 lists:
    (c_n, s_n) for n = 1..10. We force latin1 encoding so it loads cleanly
    under Python 3.
    """
    with open(EB_TEMPLATE_PKL, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    c_n = np.asarray(data[0], dtype=float)
    s_n = np.asarray(data[1], dtype=float)
    assert c_n.shape == s_n.shape == (10,), (c_n.shape, s_n.shape)
    return c_n, s_n


def make_template(H: int):
    """Return an ftperiodogram Template object truncated to the first H harmonics.

    H must be between 1 and 10 (full pickle length).
    """
    from ftperiodogram.template import Template

    c10, s10 = load_eb_coefficients()
    if not (1 <= H <= len(c10)):
        raise ValueError("H must be in 1..%d, got %d" % (len(c10), H))
    return Template(c10[:H], s10[:H])


def eb_signal(phase: np.ndarray, c_n: np.ndarray, s_n: np.ndarray) -> np.ndarray:
    """Evaluate the *unnormalized* truncated Fourier series at given phases.

    Used for injecting a signal into simulated lightcurves before adding noise.
    Note that the ``Template`` class normalizes the coefficients to unit L^2
    norm internally; for injection we want the raw shape with its natural
    peak-to-peak amplitude (the original HAT lightcurve's normalization),
    which we then rescale to the requested amplitude.

    The ``np.errstate`` wrapper suppresses spurious BLAS-level matmul
    warnings emitted on some platforms with numpy 2.x; the output is
    finite and correct.
    """
    phase = np.asarray(phase, dtype=float)
    c = np.asarray(c_n, dtype=float)
    s = np.asarray(s_n, dtype=float)
    n = np.arange(1, len(c) + 1)
    theta = 2 * np.pi * (phase[:, None] * n[None, :])  # shape (Nphase, H)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.cos(theta) @ c + np.sin(theta) @ s


# ----------------------------------------------------------------------------
# Simulated data
# ----------------------------------------------------------------------------

def simulate_lightcurve(
    rng: np.random.Generator,
    *,
    n_obs: int = 100,
    baseline: float = 10.0,
    true_freq: float = 3.7,
    amplitude: float = 1.0,
    sigma: float = 0.4,
    c_n: np.ndarray | None = None,
    s_n: np.ndarray | None = None,
):
    """Generate one realization of simulated EB lightcurve data.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator; pass with a fixed seed for reproducibility.
    n_obs : int
        Number of observations.
    baseline : float
        Total baseline in time units (days).
    true_freq : float
        Injected signal frequency in cycles per time unit.
    amplitude : float
        Peak-to-peak amplitude of the injected signal (after normalization).
        The shape is taken from the EB template and rescaled to this range.
    sigma : float
        Standard deviation of the additive Gaussian noise.
    c_n, s_n : array_like, optional
        Truncated Fourier coefficients of the injection template (default:
        the 10-harmonic EB template loaded from disk).

    Returns
    -------
    t : ndarray
        Sorted observation times in [0, baseline).
    y : ndarray
        Observations with signal + Gaussian noise.
    dy : ndarray
        Per-point uncertainties (all equal to sigma here).
    """
    if c_n is None or s_n is None:
        c_n, s_n = load_eb_coefficients()
    c_n = np.asarray(c_n)
    s_n = np.asarray(s_n)

    # uniformly-random observation times
    t = np.sort(rng.uniform(0, baseline, n_obs))
    phase = (true_freq * t) % 1.0

    raw = eb_signal(phase, c_n, s_n)
    ptp = raw.max() - raw.min()
    if ptp > 0:
        raw = raw * (amplitude / ptp)

    y = raw + rng.normal(0.0, sigma, size=n_obs)
    dy = np.full(n_obs, sigma)
    return t, y, dy


# ----------------------------------------------------------------------------
# Periodogram helpers
# ----------------------------------------------------------------------------

def autofrequency(t: np.ndarray, *, samples_per_peak: int, f_min: float, f_max: float):
    """Build a regular frequency grid bounded by [f_min, f_max].

    Per ``feedback_nyquist_irregular.md`` we set search bounds by explicit
    [f_min, f_max] rather than via a nyquist_factor.
    """
    baseline = float(t.max() - t.min())
    df = 1.0 / (baseline * samples_per_peak)
    f_min = max(f_min, df)
    Nf = int(np.ceil((f_max - f_min) / df))
    return f_min + df * np.arange(Nf + 1)


def mhls_periodogram(t, y, dy, freq, nharmonics):
    """Multi-harmonic Lomb-Scargle periodogram via gatspy.

    Returns the same fractional explained-variance metric used by FTP:
        P(f) = 1 - chi2(f) / chi2_0
    """
    from gatspy.periodic import LombScargle

    model = LombScargle(Nterms=nharmonics, fit_period=False)
    model.fit(np.asarray(t), np.asarray(y), np.asarray(dy))
    periods = 1.0 / np.asarray(freq)
    return model.periodogram(periods)


def bls_periodogram(t, y, dy, freq, qmin=0.01, qmax=0.10, n_q=8):
    """Simple Box Least Squares periodogram.

    Implements the original Kovacs 2002 algorithm:
        For each trial frequency f and each transit duration fraction q,
        slide a box and find the (i1, i2) bin pair that maximises

            SR = sqrt( s^2 / (r * (1-r)) )

        where r is the fraction of in-transit weight and s is the weighted
        sum of (y - ybar) in-transit. The returned SR is normalized to
        the [0, 1] FTP power scale:

            P(f) = SR(f)^2 / sum_n w_n (y_n - ybar)^2

        which equals 1 - chi2(f) / chi2_0 for the best (i1, i2) at this f.

    A small ``q`` grid is used (logarithmically spaced from qmin to qmax).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if dy is None:
        w = np.ones_like(y)
    else:
        w = 1.0 / np.asarray(dy, dtype=float) ** 2
    w = w / w.sum()
    ybar = float(np.dot(w, y))
    yres = y - ybar
    chi2_0 = float(np.dot(w, yres ** 2))

    freq = np.asarray(freq, dtype=float)
    qs = np.geomspace(qmin, qmax, n_q)

    # Bin grid in phase space (per Kovacs 2002).
    # Use ~200 phase bins; for each freq, fold and accumulate weighted sums.
    n_bins = 200
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    out = np.empty_like(freq)
    for k, f in enumerate(freq):
        phase = (t * f) % 1.0
        # bin the weighted residuals
        idx = np.minimum((phase * n_bins).astype(int), n_bins - 1)
        w_bin = np.bincount(idx, weights=w, minlength=n_bins)
        s_bin = np.bincount(idx, weights=w * yres, minlength=n_bins)

        # cumulative sums for fast box sweeping (double up for wraparound)
        w2 = np.concatenate([w_bin, w_bin])
        s2 = np.concatenate([s_bin, s_bin])
        cw = np.concatenate([[0.0], np.cumsum(w2)])
        cs = np.concatenate([[0.0], np.cumsum(s2)])

        best = 0.0
        for q in qs:
            box_len = max(1, int(round(q * n_bins)))
            r = cw[box_len:box_len + n_bins] - cw[:n_bins]
            ssum = cs[box_len:box_len + n_bins] - cs[:n_bins]
            mask = (r > 1e-12) & (r < 1.0 - 1e-12)
            if not np.any(mask):
                continue
            denom = r[mask] * (1.0 - r[mask])
            sr2 = (ssum[mask] ** 2) / denom
            cand = float(sr2.max())
            if cand > best:
                best = cand
        out[k] = best / chi2_0

    return out


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson linear correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = x.mean()
    my = y.mean()
    sx = x - mx
    sy = y - my
    num = float(np.dot(sx, sy))
    den = float(np.sqrt(np.dot(sx, sx) * np.dot(sy, sy)))
    if den == 0:
        return float("nan")
    return num / den


def save_fig(fig, basename: str) -> Tuple[str, str]:
    """Save *fig* to PLOTS_DIR/<basename>.pdf and .png. Returns (pdf, png)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pdf = os.path.join(PLOTS_DIR, basename + ".pdf")
    png = os.path.join(PLOTS_DIR, basename + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    return pdf, png
