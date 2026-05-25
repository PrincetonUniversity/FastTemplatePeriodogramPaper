"""Tier B: period-recovery rate on synthetic noisy light curves.

For each template, we compare two periodograms over the same trial
light curves:

  * **FTP@H** — Fast Template Periodogram using the H-truncated Fourier
    expansion of the analytic template, where ``H`` is the
    ``min_h_at_95%`` value from the Tier A fidelity result.
  * **LS-equivalent (FTP@1)** — same FTP code, but with only the
    fundamental harmonic.  This is identical to standard Lomb-Scargle
    on a sinusoidal template; using FTP for both isolates the effect
    of template shape from the choice of estimator.

Recovery is declared when the periodogram peak frequency satisfies
``|f_peak - f_true| / f_true < REL_TOL``.

Synthetic light curves use N_OBS irregular samples drawn from
[0, T_OBS), with i.i.d. Gaussian noise scaled to a per-trial
peak-to-peak SNR.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from templates import TEMPLATES, PRETTY
from fidelity import fourier_coefficients

from ftperiodogram.template import Template
from ftperiodogram.modeler import FastTemplatePeriodogram


# Survey-like sampling
N_OBS = 100
T_OBS = 100.0  # days (arbitrary unit)

# Per-template true periods (days).  Chosen well inside the search grid.
TRUE_PERIODS = {
    "rrab": 0.55,
    "rrc": 0.35,
    "transit": 3.10,
    "eb_detached": 5.20,
    "ellipsoidal_beaming": 1.40,
}

SNR_LEVELS = (3.0, 7.0, 15.0)
N_TRIALS = 30
REL_TOL = 5e-3  # 0.5 percent; loose enough for the sparse-sampling regime
RNG_SEED = 20260525

# Explicit frequency window for the search.  Nyquist does not apply to
# irregularly sampled data, so we set the upper bound by the physical
# regime (RR Lyrae, transits, EBs all have P >~ 0.2 day).
MIN_FREQ = 0.01  # cycles / day
MAX_FREQ = 5.0   # cycles / day  (covers all TRUE_PERIODS with margin)
SAMPLES_PER_PEAK = 10


def synth_lightcurve(template_fn, period: float, snr: float,
                     rng: np.random.Generator):
    """Sparse, irregularly-sampled noisy LC for one trial."""
    t = np.sort(rng.uniform(0.0, T_OBS, size=N_OBS))
    phase = (t / period) % 1.0
    y_clean = template_fn(phase)
    ptp = float(np.ptp(y_clean))
    sigma = ptp / snr
    y = y_clean + rng.normal(scale=sigma, size=N_OBS)
    dy = np.full(N_OBS, sigma)
    return t, y, dy


def ftp_periodogram(t, y, dy, c_n, s_n):
    template = Template(np.asarray(c_n), np.asarray(s_n))
    model = FastTemplatePeriodogram(template=template)
    model.fit(t, y, dy)
    freqs, power = model.autopower(samples_per_peak=SAMPLES_PER_PEAK,
                                   minimum_frequency=MIN_FREQ,
                                   maximum_frequency=MAX_FREQ)
    return freqs, np.asarray(power)


def recovered(f_peak: float, f_true: float) -> bool:
    return abs(f_peak - f_true) / f_true < REL_TOL


def run():
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    out: dict = {}

    for key, fn in TEMPLATES.items():
        period = TRUE_PERIODS[key]
        f_true = 1.0 / period

        # Tier A result: load min_h_at_95
        with open(Path(__file__).parent / "results" / "fidelity.pkl", "rb") as fp:
            fid = pickle.load(fp)
        h_template = fid[key]["min_h_at"][0.95]
        if h_template is None:
            h_template = 30  # cap if 95% never reached (shouldn't happen here)

        c_full, s_full = fourier_coefficients(fn, h=max(h_template, 1))
        c_1, s_1 = fourier_coefficients(fn, h=1)

        per_snr: dict = {}
        for snr in SNR_LEVELS:
            ftp_hits = 0
            ls_hits = 0
            for trial in range(N_TRIALS):
                t, y, dy = synth_lightcurve(fn, period, snr, rng)

                freqs, power = ftp_periodogram(t, y, dy, c_full, s_full)
                f_peak_ftp = float(freqs[np.argmax(power)])
                if recovered(f_peak_ftp, f_true):
                    ftp_hits += 1

                freqs_ls, power_ls = ftp_periodogram(t, y, dy, c_1, s_1)
                f_peak_ls = float(freqs_ls[np.argmax(power_ls)])
                if recovered(f_peak_ls, f_true):
                    ls_hits += 1

            per_snr[snr] = {
                "ftp_rate": ftp_hits / N_TRIALS,
                "ls_rate": ls_hits / N_TRIALS,
            }
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] {PRETTY[key]:>22s} SNR={snr:>4.1f} "
                  f"FTP@H={h_template:>2d}: {ftp_hits}/{N_TRIALS}  "
                  f"FTP@1: {ls_hits}/{N_TRIALS}", flush=True)

        out[key] = {
            "pretty": PRETTY[key],
            "h_used": h_template,
            "period": period,
            "by_snr": per_snr,
        }

        # Incremental save in case of interruption
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "recovery.pkl", "wb") as fp:
            pickle.dump(out, fp)

    total = time.time() - t0
    print(f"\nfinished in {total:.1f}s; results at "
          f"{Path(__file__).parent / 'results' / 'recovery.pkl'}")
    return out


if __name__ == "__main__":
    run()
