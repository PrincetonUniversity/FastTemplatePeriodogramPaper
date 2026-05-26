"""Figure 3 of paper_v5: ``correlation_with_large_H.pdf``.

Multi-panel scatter (one row, four columns).  Each panel compares the
*peak* FTP@H=10 value to the *peak* FTP@H=h value for one fixed
h in {1, 2, 5, 9}, computed over the same MC realizations as Fig 2.

Following the task spec, the x-axis is the H=10 result (high-H
"reference") and the y-axis is the lower H=h result.  Each panel
annotates Pearson R for that pair.

Reviewer comment addressed: JH.30 -- we count and tabulate
realizations where the low-H peak exceeds the H=10 peak, and document
the empirical fraction in README.md.

Run with

    .../FastTemplatePeriodogram/.venv/bin/python \
        scripts/figs_v5/fig3_corrwhighh.py [--n-realizations N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import matplotlib.pyplot as plt

from common import (
    COL_DIAG,
    COL_FTP,
    autofrequency,
    make_template,
    pearson_r,
    save_fig,
    setup_mpl,
    simulate_lightcurve,
)


H_REF = 10
H_LIST = (1, 2, 5, 9)


def _single_realization(sub_seed: int):
    """Compute peak FTP power for H in H_LIST + H_REF for one realization."""
    from ftperiodogram.modeler import FastTemplatePeriodogram

    rng = np.random.default_rng(sub_seed)
    true_freq = float(rng.uniform(1.0, 8.0))

    t, y, dy = simulate_lightcurve(
        rng,
        n_obs=80,
        baseline=10.0,
        true_freq=true_freq,
        amplitude=1.0,
        sigma=0.4,
    )

    freq = autofrequency(t, samples_per_peak=5, f_min=0.5, f_max=15.0)

    peaks = {}
    for H in (*H_LIST, H_REF):
        modeler = FastTemplatePeriodogram(template=make_template(H))
        modeler.fit(t, y, dy)
        p = modeler.power(freq, save_best_model=False)
        peaks[H] = float(p.max())
    return peaks


def run_realizations(n_realizations: int, seed: int = 20260525, n_workers: int = 0):
    master_rng = np.random.default_rng(seed)
    sub_seeds = master_rng.integers(0, 2**31 - 1, size=n_realizations)

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    print(f"  using {n_workers} worker(s)")

    if n_workers == 1:
        results = (_single_realization(int(s)) for s in sub_seeds)
    else:
        pool = mp.get_context("fork").Pool(n_workers)
        results = pool.imap(_single_realization, [int(s) for s in sub_seeds])

    peaks = {H: np.zeros(n_realizations) for H in (*H_LIST, H_REF)}
    try:
        for i, row in enumerate(results):
            for H in (*H_LIST, H_REF):
                peaks[H][i] = row[H]
            if (i + 1) % 25 == 0 or i + 1 == n_realizations:
                print(f"  realization {i+1}/{n_realizations} "
                      f"P10={peaks[H_REF][i]:.3f}  P5={peaks[5][i]:.3f}")
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    return peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=300)
    parser.add_argument("--n-workers", type=int, default=0)
    args = parser.parse_args()

    setup_mpl()

    print(f"running {args.n_realizations} realizations...")
    peaks = run_realizations(args.n_realizations, n_workers=args.n_workers)
    p10 = peaks[H_REF]

    print()
    print("JH.30 summary (fraction of realizations with P_FTP(H=h) > "
          "P_FTP(H=10)):")
    for H in H_LIST:
        ph = peaks[H]
        n_above = int((ph > p10 + 1e-6).sum())
        n_above_meaningful = int((ph > p10 + 0.001).sum())
        R = pearson_r(p10, ph)
        print(f"  H={H:>2d}  R={R:.3f}  "
              f"P_h > P_10 in {n_above}/{args.n_realizations} "
              f"({100.0*n_above/args.n_realizations:.1f}%)  "
              f"meaningful (> 0.001 above): {n_above_meaningful}")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(
        1, len(H_LIST), figsize=(2.6 * len(H_LIST) + 0.3, 2.7),
        sharex=True, sharey=True,
        gridspec_kw=dict(wspace=0.15),
    )

    for ax, H in zip(axes, H_LIST):
        ph = peaks[H]
        R = pearson_r(p10, ph)
        ax.plot([0, 1], [0, 1], color=COL_DIAG, ls="--", lw=0.7, zorder=1)
        ax.scatter(p10, ph, s=5, color=COL_FTP, alpha=0.85, zorder=3,
                   edgecolors="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"$P_{\mathrm{FTP}}(\omega\,|\,H=10)$")
        ax.set_title(rf"$H = {H}$", fontsize=10)
        ax.annotate(
            f"$R = {R:.3f}$",
            xy=(0.05, 0.95), xycoords="axes fraction",
            ha="left", va="top", fontsize=9,
        )

    axes[0].set_ylabel(r"$P_{\mathrm{FTP}}(\omega\,|\,H=h)$")

    pdf, png = save_fig(fig, "correlation_with_large_H")
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
