"""
Fresh, self-contained Lomb-Scargle timing micro-benchmark for paper_v5 Sec 1.3
(Kipping K.10: give absolute LS computation times, not just scaling).

Times the two ways of computing a Lomb-Scargle periodogram that Sec 1.3
discusses, on irregularly-sampled data over an EXPLICIT [f_min, f_max] grid
(no Nyquist-based bounds, per project convention):

  - direct  : O(N_f * N_obs)   -- gatspy.periodic.LombScargle
  - NFFT/FFT: O(N_f log N_f)    -- gatspy.periodic.LombScargleFast (Press & Rybicki extirpolation)

Reports per-source wall-clock and per-(frequency x observation) cost, plus the
host/library versions, so the numbers quoted in Sec 1.3 are reproducible and can
be refreshed on submission hardware. Does NOT use the deprecated generate_plots.py.

Run (from the paper repo, with the FTP venv that has gatspy + nfft):
    python scripts/timing_v5/ls_timing.py
"""
import json
import platform
import time
import os

import numpy as np
from gatspy.periodic import LombScargle, LombScargleFast

RSEED = 20260528
N_OBS = [50, 200, 1000]      # representative source sizes
N_FREQ = 10000               # trial frequencies
T_BASELINE = 100.0           # days
F_MIN, F_MAX = 0.01, 10.0    # cyc/day -- explicit physical bounds, NOT Nyquist
N_REPEAT = 5                 # best-of-N timing


def make_lightcurve(n_obs, rng):
    t = np.sort(rng.uniform(0, T_BASELINE, n_obs))
    y = np.sin(2 * np.pi * 1.23 * t) + 0.1 * rng.standard_normal(n_obs)
    dy = 0.1 * np.ones(n_obs)
    return t, y, dy


def best_time(fn, repeat=N_REPEAT):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(RSEED)
    f0, df, Nf = F_MIN, (F_MAX - F_MIN) / N_FREQ, N_FREQ

    rows = []
    for n_obs in N_OBS:
        t, y, dy = make_lightcurve(n_obs, rng)

        fast = LombScargleFast(silence_warnings=True).fit(t, y, dy)
        slow = LombScargle().fit(t, y, dy)

        t_fast = best_time(lambda: fast.score_frequency_grid(f0, df, Nf))
        t_slow = best_time(lambda: slow.score_frequency_grid(f0, df, Nf))

        rows.append({
            "n_obs": n_obs, "n_freq": Nf,
            "t_fast_s": t_fast, "t_slow_s": t_slow,
            "fast_us_per_freq": 1e6 * t_fast / Nf,
            "slow_us_per_freq": 1e6 * t_slow / Nf,
        })

    info = {
        "host": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "grid": {"f_min": F_MIN, "f_max": F_MAX, "n_freq": N_FREQ},
        "rows": rows,
    }

    print(json.dumps(info, indent=2))
    print("\n--- summary (best of %d) ---" % N_REPEAT)
    for r in rows:
        print("N_obs=%5d  N_f=%d :  NFFT %.1f ms (%.3f us/freq)   direct %.1f ms (%.3f us/freq)"
              % (r["n_obs"], r["n_freq"],
                 1e3 * r["t_fast_s"], r["fast_us_per_freq"],
                 1e3 * r["t_slow_s"], r["slow_us_per_freq"]))

    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ls_timing.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
