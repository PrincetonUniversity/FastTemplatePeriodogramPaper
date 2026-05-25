"""Tier A: cumulative L^2-explained-variance vs harmonic order H.

For a real periodic template y(phi) sampled at N equally-spaced phases,
write the Fourier expansion

    y(phi) = c_0 + sum_{n=1}^{N/2} (c_n cos(2 pi n phi) + s_n sin(2 pi n phi)).

The H-truncated approximation y_H is the optimal L^2 approximation among
trigonometric polynomials of degree H, and the explained-variance ratio
is

    F(H) = sum_{n=1}^H (c_n^2 + s_n^2) / sum_{n=1}^{N/2} (c_n^2 + s_n^2).

The DC component c_0 is excluded because FTP treats the DC offset as a
free parameter that is fit per-frequency.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from templates import TEMPLATES, PRETTY


N_PHASE = 8192
H_MAX = 50
THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


def cumulative_fidelity(template_fn, n_phase: int = N_PHASE,
                        h_max: int = H_MAX):
    """Cumulative explained-variance fraction at H = 1 .. h_max.

    Returns
    -------
    h : (h_max,) int array, harmonic indices 1..h_max
    fid : (h_max,) float array, cumulative L^2 explained-variance fraction
    """
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    y = template_fn(phase)
    y = y - y.mean()  # drop DC
    Y = np.fft.rfft(y)
    # |Y[k]|^2 is proportional to the L^2 energy at harmonic k
    energy = np.abs(Y[1:]) ** 2  # drop DC bin
    total = energy.sum()
    if total <= 0.0:
        raise ValueError("template has zero AC energy")
    cum = np.cumsum(energy[:h_max]) / total
    h = np.arange(1, h_max + 1)
    return h, cum


def min_h_for(cum: np.ndarray, threshold: float) -> int | None:
    """Smallest H such that cum[H-1] >= threshold; None if never reached."""
    hits = np.where(cum >= threshold)[0]
    return int(hits[0] + 1) if hits.size > 0 else None


def fourier_coefficients(template_fn, h: int, n_phase: int = N_PHASE):
    """Return (c_n, s_n) arrays for n = 1..h with y ~ c_0 + sum c_n cos + s_n sin."""
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    y = template_fn(phase)
    Y = np.fft.rfft(y)
    # numpy.fft.rfft of length-N real array: Y[k] = sum_j y[j] exp(-2 pi i j k / N)
    # Inverse: y[j] = (1/N) sum_k Y[k] exp(2 pi i j k / N) for real spectrum
    # Real coefficients: c_n = 2 Re(Y[n]) / N, s_n = -2 Im(Y[n]) / N for n >= 1.
    n_arr = np.arange(1, h + 1)
    coeffs = 2.0 * Y[n_arr] / n_phase
    c_n = coeffs.real
    s_n = -coeffs.imag
    return c_n, s_n


def run():
    out: dict = {}
    for key, fn in TEMPLATES.items():
        h, cum = cumulative_fidelity(fn)
        out[key] = {
            "pretty": PRETTY[key],
            "h": h,
            "cum_fidelity": cum,
            "min_h_at": {t: min_h_for(cum, t) for t in THRESHOLDS},
        }
        print(f"{PRETTY[key]:>25s}: " +
              " ".join(f"{t*100:>5.1f}%->H={out[key]['min_h_at'][t]!s:>4s}"
                       for t in THRESHOLDS))

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "fidelity.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\nsaved {results_dir / 'fidelity.pkl'}")
    return out


if __name__ == "__main__":
    run()
