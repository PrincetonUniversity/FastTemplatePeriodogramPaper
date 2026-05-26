"""Figure 2 of paper_v5: ``correlation_with_nonlinopt.pdf``.

Scatter plot of the *peak* periodogram values from independent MC
realizations of simulated EB lightcurves, FTP@H=10 (x-axis) versus the
slow non-linear-optimization template fitter (``SlowTemplatePeriodogram``,
y-axis). Pearson R is annotated. The expected pattern is points on or
below the diagonal: the slow fitter occasionally finds only a local
chi^2 minimum and reports a smaller P value than the FTP.

Reviewer comments addressed:
- JH.29: we explicitly count and tabulate the (rare) realizations where
  P_slow > P_FTP and document the size of the excess in
  scripts/figs_v5/README.md.
- JVP.7: scatter points are color-coded by whether *both* algorithms
  recovered the injected frequency to within RECOVERY_TOL, with the
  count annotated in-panel. This answers JVP's concern that two
  algorithms with high Pearson R might just be agreeing on the wrong
  answer.

Run with

    .../FastTemplatePeriodogram/.venv/bin/python \
        scripts/figs_v5/fig2_corrwgats.py [--n-realizations N]
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
    COL_MISS,
    RECOVERY_TOL,
    autofrequency,
    make_template,
    pearson_r,
    save_fig,
    setup_mpl,
    simulate_lightcurve,
)


# module-level so multiprocessing can pickle it
def _single_realization(sub_seed: int):
    from ftperiodogram.modeler import (
        FastTemplatePeriodogram,
        SlowTemplatePeriodogram,
    )

    template = make_template(10)
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

    freq = autofrequency(
        t, samples_per_peak=5, f_min=0.5, f_max=15.0,
    )

    # FTP@H=10
    fast = FastTemplatePeriodogram(template=template)
    fast.fit(t, y, dy)
    pf = fast.power(freq, save_best_model=False)

    # Slow fitter on the same grid.
    slow = SlowTemplatePeriodogram(template=template, nguesses=5)
    np.random.seed(sub_seed)
    slow.fit(t, y, dy)
    ps = slow.power(freq)

    # also store argmax mismatch (in units of bins)
    i_fast = int(np.argmax(pf))
    i_slow = int(np.argmax(ps))
    return (float(pf.max()), float(ps.max()), i_fast, i_slow,
            float(freq[i_fast]), float(freq[i_slow]), true_freq)


def run_realizations(n_realizations: int, seed: int = 20260525, n_workers: int = 0):
    """Run *n_realizations* MC realizations of the EB-injection experiment.

    For each realization we compute the FTP periodogram on a frequency
    grid, then evaluate the slow fitter on the *same* grid. We record
    the maximum power from each, and the bin indices of the peaks so
    we can quantify any disagreement (JH.29 hypothesis check).
    """
    master_rng = np.random.default_rng(seed)
    sub_seeds = master_rng.integers(0, 2**31 - 1, size=n_realizations)

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    print(f"  using {n_workers} worker(s)")

    p_ftp = np.zeros(n_realizations)
    p_slow = np.zeros(n_realizations)
    i_ftp = np.zeros(n_realizations, dtype=int)
    i_slow = np.zeros(n_realizations, dtype=int)
    f_ftp = np.zeros(n_realizations)
    f_slow = np.zeros(n_realizations)
    f_true = np.zeros(n_realizations)

    if n_workers == 1:
        results = (_single_realization(int(s)) for s in sub_seeds)
    else:
        pool = mp.get_context("fork").Pool(n_workers)
        results = pool.imap(_single_realization, [int(s) for s in sub_seeds])

    try:
        for i, row in enumerate(results):
            (p_ftp[i], p_slow[i], i_ftp[i], i_slow[i],
             f_ftp[i], f_slow[i], f_true[i]) = row
            if (i + 1) % 25 == 0 or i + 1 == n_realizations:
                print(f"  realization {i+1}/{n_realizations} "
                      f"P_FTP={p_ftp[i]:.3f}  P_slow={p_slow[i]:.3f}")
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    return dict(
        p_ftp=p_ftp, p_slow=p_slow,
        i_ftp=i_ftp, i_slow=i_slow,
        f_ftp=f_ftp, f_slow=f_slow, f_true=f_true,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=300,
                        help="Number of MC realizations (default 300).")
    parser.add_argument("--n-workers", type=int, default=0,
                        help="Number of worker processes (0 = auto).")
    args = parser.parse_args()

    setup_mpl()

    print(f"running {args.n_realizations} realizations...")
    out = run_realizations(args.n_realizations, n_workers=args.n_workers)
    p_ftp = out["p_ftp"]
    p_slow = out["p_slow"]

    R = pearson_r(p_ftp, p_slow)
    print(f"Pearson R = {R:.3f}")

    # JH.29 quantification
    excess = p_slow - p_ftp
    n_above_tiny = int((excess > 1e-6).sum())
    n_above_pct = int((excess > 0.001).sum())  # > 0.1 percentage point
    n_bin_diff = int((out["i_ftp"] != out["i_slow"]).sum())
    if n_above_tiny > 0:
        max_excess = float(excess.max())
        med_excess = float(np.median(excess[excess > 1e-6]))
        print(f"JH.29: {n_above_tiny}/{args.n_realizations} realizations "
              f"({100.0 * n_above_tiny / args.n_realizations:.1f}%) have "
              f"P_slow > P_FTP at the per-realization peak.")
        print(f"       {n_above_pct} of those exceed by > 0.001 in power.")
        print(f"       max excess = {max_excess:.4f}, "
              f"median (when positive) = {med_excess:.4f}.")
        print(f"       {n_bin_diff} realizations have differing argmax "
              f"frequency bins.")
    else:
        print(f"JH.29: 0/{args.n_realizations} realizations have "
              f"P_slow > P_FTP. The FTP dominates the slow fitter at every "
              f"realization, consistent with the caption's framing.")

    # ---------------------------------------------------------------
    # JVP.7: classify each realization by true-frequency recovery
    # ---------------------------------------------------------------
    f_ftp = out["f_ftp"]
    f_slow = out["f_slow"]
    f_true = out["f_true"]
    ftp_ok = np.abs(f_ftp - f_true) / f_true < RECOVERY_TOL
    slow_ok = np.abs(f_slow - f_true) / f_true < RECOVERY_TOL
    both_ok = ftp_ok & slow_ok
    n_both = int(both_ok.sum())
    n_total = both_ok.size
    print(f"JVP.7: true-freq recovery (|df/f| < {RECOVERY_TOL}):  "
          f"FTP {int(ftp_ok.sum())}/{n_total}, "
          f"slow {int(slow_ok.sum())}/{n_total}, "
          f"both {n_both}/{n_total}.")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    # diagonal
    ax.plot([0, 1], [0, 1], color=COL_DIAG, ls="--", lw=0.7, zorder=1)
    # both recovered: filled black
    ax.scatter(p_ftp[both_ok], p_slow[both_ok], s=6, color=COL_FTP,
               alpha=0.85, zorder=3, edgecolors="none",
               label="both recover $f_{\\rm true}$")
    # at least one missed: hollow red, more visually salient
    miss = ~both_ok
    if miss.any():
        ax.scatter(p_ftp[miss], p_slow[miss], s=18, facecolor="none",
                   edgecolors=COL_MISS, linewidth=0.8, zorder=4,
                   label=r"$\geq 1$ misses $f_{\rm true}$")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"$P_{\mathrm{FTP}}(\omega\,|\,H=10)$")
    ax.set_ylabel(r"$P_{\mathrm{slow}}(\omega)$")
    ax.annotate(
        f"$R = {R:.3f}$\n"
        f"both recover $f_{{\\rm true}}$: {n_both}/{n_total}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=9,
    )
    if miss.any():
        ax.legend(loc="lower right", fontsize=8, borderaxespad=0.3,
                  handlelength=1.0, scatterpoints=1)

    pdf, png = save_fig(fig, "correlation_with_nonlinopt")
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
