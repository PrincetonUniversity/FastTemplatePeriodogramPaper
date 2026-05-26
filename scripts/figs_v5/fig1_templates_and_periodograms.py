"""Figure 1 of paper_v5: ``templates_and_periodograms.pdf``.

Rows are H = 1, 2, 5, 9, 10.  Left column: phase-folded simulated EB
lightcurve with the H-harmonic FTP best-fit overlaid.  Right column:
the FTP periodogram (black), with the vertical dotted guide marking
the true injected frequency.  H = 2, 5, 9 also overlay the
multi-harmonic Lomb-Scargle periodogram (blue).  H = 10 also overlays
a Box Least Squares periodogram (purple).

Run from the repo root with

    .../FastTemplatePeriodogram/.venv/bin/python \
        scripts/figs_v5/fig1_templates_and_periodograms.py

Addresses reviewer comments:
- JH.31/L.6: ensure proper minus glyph in the d^{-1} unit on the freq axis.
- JVP.5(b): label the FTP curve "Fast Template Method" in the periodogram
  legend so the black curve is identified alongside the MHLS / BLS overlays.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    COL_BLS,
    COL_DATA,
    COL_FTP,
    COL_MHLS,
    COL_TRUEF,
    LW_MED,
    LW_THIN,
    MS_PTS,
    autofrequency,
    bls_periodogram,
    load_eb_coefficients,
    make_template,
    mhls_periodogram,
    pearson_r,  # noqa: F401  (kept for parity with other figs)
    save_fig,
    setup_mpl,
    simulate_lightcurve,
)


def main() -> None:
    setup_mpl()

    # Single realization, single random seed.
    seed = 20260525
    rng = np.random.default_rng(seed)

    # Signal parameters chosen so the FTP@H=10 peak power is around 0.4-0.6
    # (matches the visual range of the previous published figure).
    true_freq = 3.1       # cycles per day
    n_obs = 200
    baseline = 10.0
    amplitude = 1.0
    sigma = 0.18
    t, y, dy = simulate_lightcurve(
        rng,
        n_obs=n_obs,
        baseline=baseline,
        true_freq=true_freq,
        amplitude=amplitude,
        sigma=sigma,
    )

    # Frequency grid for the periodograms. Tight enough to resolve H=10 peaks
    # but coarse enough to render cleanly in print.
    f_min, f_max = 0.5, 30.0
    freq = autofrequency(t, samples_per_peak=12, f_min=f_min, f_max=f_max)

    # Rows
    Hs = [1, 2, 5, 9, 10]

    # Compute FTP for each H. Save best-fit model at the highest power
    # for the lightcurve panel.
    from ftperiodogram.modeler import FastTemplatePeriodogram

    ftp_power: dict[int, np.ndarray] = {}
    ftp_best_model = {}
    for H in Hs:
        modeler = FastTemplatePeriodogram(template=make_template(H))
        modeler.fit(t, y, dy)
        # Use power() on the shared grid so FTP/MHLS/BLS are on identical f.
        p = modeler.power(freq)
        ftp_power[H] = (freq, p)
        ftp_best_model[H] = modeler.best_model

    # MHLS for H in {2, 5, 9} (gatspy LombScargle Nterms=H)
    mhls_power: dict[int, np.ndarray] = {}
    for H in (2, 5, 9):
        mhls_power[H] = mhls_periodogram(t, y, dy, freq, nharmonics=H)

    # BLS for the bottom (H=10) row, on the same frequency grid.
    bls_power = bls_periodogram(t, y, dy, freq)

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    nrows = len(Hs)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(8.0, 1.3 * nrows + 0.8),
        gridspec_kw=dict(width_ratios=[1.0, 2.4], wspace=0.18, hspace=0.35),
    )

    # phase grid for the template overlay
    phase_grid = np.linspace(0, 1, 400)

    for row, H in enumerate(Hs):
        ax_lc = axes[row, 0]
        ax_pg = axes[row, 1]

        # ------- Left column: phase-folded lightcurve + best fit ----
        phase = (true_freq * t) % 1.0
        ax_lc.errorbar(
            phase, y, yerr=dy,
            fmt=".", ms=MS_PTS * 1.5, lw=0.5, capsize=0,
            color=COL_DATA, alpha=0.85, zorder=2,
        )

        # Overlay the best-fit template at the *true* freq (so the
        # eye sees the model overlapping the data; using the FTP's
        # peak freq would generally agree to within df).
        model = ftp_best_model[H]
        t_model = np.linspace(0, 1.0 / true_freq, 400)
        # set frequency to true freq for a clean overlay
        model_fixed = type(model)(
            template=model.template,
            frequency=true_freq,
            parameters=model.parameters,
        )
        y_model = model_fixed(t_model)
        ph_model = (true_freq * t_model) % 1.0
        order = np.argsort(ph_model)
        ax_lc.plot(
            ph_model[order], y_model[order],
            color=COL_FTP, lw=LW_MED, zorder=3,
        )

        ax_lc.set_xlim(0, 1)
        ax_lc.set_xticks([0, 0.5, 1.0])
        ax_lc.set_yticks([])
        ax_lc.annotate(
            f"H = {H}",
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-2, -2), textcoords="offset points",
            ha="right", va="top", fontsize=10,
        )
        if row == 0:
            ax_lc.set_title("Template fit", fontsize=10)
        if row == nrows - 1:
            ax_lc.set_xlabel("Phase")
        else:
            ax_lc.set_xticklabels([])

        # ------- Right column: periodograms -------------------------
        f, p = ftp_power[H]
        # Plot MHLS first (blue, behind FTP) for H in {2,5,9}
        if H in (2, 5, 9):
            ax_pg.plot(
                freq, mhls_power[H],
                color=COL_MHLS, lw=0.4, alpha=0.75, zorder=2,
                label="Multi-harmonic Lomb-Scargle",
            )
        if H == 10:
            ax_pg.plot(
                freq, bls_power,
                color=COL_BLS, lw=0.5, alpha=0.85, zorder=2,
                label="Box Least Squares",
            )

        ax_pg.plot(f, p, color=COL_FTP, lw=0.6, zorder=4,
                   label="Fast Template Method")

        # True-frequency guide (dotted)
        ax_pg.axvline(
            true_freq, color=COL_TRUEF, ls=":", lw=0.7, zorder=1,
        )

        ax_pg.set_xlim(f_min, f_max)
        # Fixed y-range so visual comparison across H is meaningful.
        ax_pg.set_ylim(-0.02, 0.75)
        ax_pg.set_yticks([])

        if row == 0:
            ax_pg.set_title("Template periodogram", fontsize=10)
        if row == nrows - 1:
            # Use mathtext so the minus is a real glyph (L.6 fix).
            ax_pg.set_xlabel(r"Frequency [d$^{-1}$]")
        else:
            ax_pg.set_xticklabels([])

        # Legend: top row introduces FTP, H=2 adds MHLS, H=10 adds BLS
        # (per JVP.5(b): "Fast Template Method" should be named, in black,
        # alongside the other algorithm labels).
        if H in (1, 2, 10):
            ax_pg.legend(loc="upper right", borderaxespad=0.3,
                         handlelength=1.5)

    pdf, png = save_fig(fig, "templates_and_periodograms")
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
