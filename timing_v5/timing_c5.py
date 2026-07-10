#!/usr/bin/env python
"""Fresh timing benchmark for the FTP performance narrative (WP C5).

Compares the default scan+polish maximizer (``method='scan'``, the default since
WP C4) against the permanent reference implementation (``method='eigvals'``, the
per-frequency polynomial-root path) and against full non-linear optimisation at
each trial frequency (:class:`SlowTemplatePeriodogram`).

Run with the code repo's venv, from this directory::

    ../../../FastTemplatePeriodogram/.venv/bin/python timing_c5.py
    ../../../FastTemplatePeriodogram/.venv/bin/python timing_c5.py --figures-only

The plain form measures everything, writes ``timing_results.json`` (all raw
timings + seeds + config) here, and regenerates
``../plots/timing_vs_nharm.{pdf,png}``, ``../plots/timing_vs_ndata.{pdf,png}``
and ``../plots/timing_vs_ndata_const_freq.{pdf,png}``.  ``--figures-only``
re-renders the figures and re-prints the SUMMARY from the saved JSON without
re-timing.  All numbers quoted in paper_v5.tex must come from this script's
printed SUMMARY block, measured on the submission machine in the same session
(WP C5 rule iv).

Timing methodology: each configuration is timed ``reps`` times and the MINIMUM
wall time is reported (least contended run).  Data are well-conditioned random-
cadence single-band light curves unless noted; the scan advantage shrinks toward
1x on sparse / deep-|MM| configurations where the exact root fallback fires (see
the multiband block and VERIFICATION.md WP C3/C3.5).
"""
import json
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ftperiodogram.template import Template
from ftperiodogram.core import template_periodogram
from ftperiodogram.modeler import SlowTemplatePeriodogram
from ftperiodogram.multiband import (multiband_template_periodogram,
                                     build_template_set)

# ---------------------------------------------------------------------------
# Fixed configuration (recorded seeds -> reproducible)
# ---------------------------------------------------------------------------
SEED = 1                 # single global RNG seed for cadence + noise
BASELINE = 250.0         # days
FMIN = 0.5               # c/d
SPP = 5                  # frequency samples per peak (sets df = 1/(baseline*spp))
NOISE = 0.02             # mag; uniform dy
PERIOD = 0.55            # injected period (days) for the template signal

N_LIST = [15, 30, 60, 125, 250, 500, 1000, 2000, 4000, 10000]
SLOW_NMAX = 250          # non-linear optimisation only feasible up to here
EIG_NMAX_CADENCE = 2000  # cap the reference path where Nf=12N grows large


def make_template(H):
    """A representative, well-conditioned non-sinusoidal template with H
    harmonics (decaying 1/n amplitudes, fixed phase pattern)."""
    n = np.arange(1, H + 1)
    return Template(np.cos(0.6 * n) / n, np.sin(0.6 * n) / n)


def grid(baseline, Nf, spp=SPP, fmin=FMIN):
    """A valid NFFT frequency grid: Nf points at integer multiples of df,
    starting at/just below fmin."""
    df = 1.0 / (baseline * spp)
    nf0 = max(1, int(np.floor(fmin / df)))
    return df * (nf0 + np.arange(Nf))


def make_data(N, rng, baseline=BASELINE):
    t = np.sort(rng.uniform(0.0, baseline, N))
    tmpl = make_template(6)
    y = tmpl(t / PERIOD) + NOISE * rng.standard_normal(N)
    dy = NOISE * np.ones(N)
    return t, y, dy


def time_call(fn, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ===========================================================================
def measure():
    results = {"config": dict(seed=SEED, baseline=BASELINE, fmin=FMIN, spp=SPP,
                              noise=NOISE, period=PERIOD)}

    # A. Single-band time vs H  (nharm figure + scan-vs-eigvals scaling in H)
    print("== A: vs H (N=200, Nf=6000) ==")
    rng = np.random.default_rng(SEED)
    A_N, A_Nf = 200, 6000
    t, y, dy = make_data(A_N, rng)
    freqs = grid(BASELINE, A_Nf)
    A = {"N": A_N, "Nf": A_Nf, "H": [], "eigvals": [], "scan": []}
    for H in range(1, 13):
        tmpl = make_template(H)
        te = time_call(lambda: template_periodogram(
            t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="eigvals"), reps=2)
        ts = time_call(lambda: template_periodogram(
            t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="scan"), reps=3)
        A["H"].append(H)
        A["eigvals"].append(te)
        A["scan"].append(ts)
        print(f"  H={H:2d}  eigvals={te:7.3f}s  scan={ts:6.3f}s  speedup={te/ts:5.1f}x")
    results["vs_H"] = A

    # D. Single-band scan-vs-eigvals at H=8 across grid size Nf
    print("== D: H=8, Nf sweep (N=300) ==")
    rng = np.random.default_rng(SEED + 7)
    D_N = 300
    t, y, dy = make_data(D_N, rng)
    tmpl = make_template(8)
    D = {"N": D_N, "H": 8, "Nf": [], "eigvals": [], "scan": []}
    for Nf in (2000, 8000, 20000):
        freqs = grid(BASELINE, Nf)
        te = time_call(lambda: template_periodogram(
            t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="eigvals"), reps=2)
        ts = time_call(lambda: template_periodogram(
            t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="scan"), reps=3)
        D["Nf"].append(Nf)
        D["eigvals"].append(te)
        D["scan"].append(ts)
        print(f"  Nf={Nf:6d}  eigvals={te:7.3f}s  scan={ts:6.3f}s  speedup={te/ts:5.1f}x")
    results["H8_vs_Nf"] = D

    # B/C. Single-band vs N_obs, two regimes
    def run_ndata(regime, H=6):
        print(f"== {regime}: vs N_obs (H={H}) ==")
        out = {"N": [], "Nf": [], "scan": [], "eigvals": [], "slow": []}
        for N in N_LIST:
            rng = np.random.default_rng(SEED + N)
            t, y, dy = make_data(N, rng)
            Nf = int(round(12 * N)) if regime == "const_cadence" else 2000
            freqs = grid(BASELINE, Nf)
            tmpl = make_template(H)
            reps = 3 if N <= 250 else 1
            ts = time_call(lambda: template_periodogram(
                t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="scan"), reps=reps)
            if regime == "const_freq" or N <= EIG_NMAX_CADENCE:
                te = time_call(lambda: template_periodogram(
                    t, y, dy, tmpl.c_n, tmpl.s_n, freqs, method="eigvals"),
                    reps=(2 if N <= 250 else 1))
            else:
                te = None
            if N <= SLOW_NMAX:
                slow = SlowTemplatePeriodogram(template=tmpl, nguesses=10).fit(t, y, dy)
                tl = time_call(lambda: slow.power(freqs), reps=1)
            else:
                tl = None
            out["N"].append(N)
            out["Nf"].append(Nf)
            out["scan"].append(ts)
            out["eigvals"].append(te)
            out["slow"].append(tl)
            se = f"{te:7.3f}s" if te is not None else "    -   "
            sl = f"{tl:8.3f}s" if tl is not None else "     -    "
            print(f"  N={N:6d} Nf={Nf:7d}  scan={ts:7.3f}s  eigvals={se}  slow={sl}")
        return out

    results["ndata_const_cadence"] = run_ndata("const_cadence")
    results["ndata_const_freq"] = run_ndata("const_freq")

    # E. Multiband shared_phase scan-vs-eigvals (config-dependence, MUST(ii))
    print("== E: multiband shared_phase (K=2, N=200, Nf=2000) ==")
    E = {"K": 2, "N": 200, "Nf": 2000, "H": [], "eigvals": [], "scan": []}
    for H in (3, 8):
        rng = np.random.default_rng(SEED + 100 + H)
        N, Nf = 200, 2000
        t = np.sort(rng.uniform(0.0, BASELINE, N))
        bands = np.array(["g", "r"])[rng.integers(0, 2, N)]
        tmpl = make_template(H)
        tdict = build_template_set(tmpl, bands)
        y = np.zeros(N)
        for b in ("g", "r"):
            m = bands == b
            y[m] = tmpl(t[m] / PERIOD)
        y = y + NOISE * rng.standard_normal(N)
        dy = NOISE * np.ones(N)
        freqs = grid(BASELINE, Nf)
        te = time_call(lambda: multiband_template_periodogram(
            t, y, bands, tdict, freqs, dy=dy, mode="shared_phase",
            method="eigvals"), reps=1)
        ts = time_call(lambda: multiband_template_periodogram(
            t, y, bands, tdict, freqs, dy=dy, mode="shared_phase",
            method="scan"), reps=2)
        E["H"].append(H)
        E["eigvals"].append(te)
        E["scan"].append(ts)
        print(f"  H={H} K=2  eigvals={te:7.3f}s  scan={ts:6.3f}s  speedup={te/ts:5.1f}x")
    results["multiband_shared_phase"] = E

    with open("timing_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


# ===========================================================================
# Figures
# ===========================================================================
BLUE, RED, GREY = "#1f4e79", "#c0392b", "#7f8c8d"


def make_figures(results):
    # --- Fig 1: timing_vs_nharm (per-frequency time vs H) ------------------
    A = results["vs_H"]
    H_arr = np.array(A["H"], float)
    eig_pf = np.array(A["eigvals"]) / A["Nf"] * 1e3   # ms per trial frequency
    scan_pf = np.array(A["scan"]) / A["Nf"] * 1e3
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.loglog(H_arr, eig_pf, "o-", color=RED, label="eigvals (reference)")
    ax.loglog(H_arr, scan_pf, "s-", color=BLUE, label="scan (default)")
    # short asymptotic guide segments over the high-H regime only (H>=6),
    # where the per-frequency maximiser -- not the shared NFFT floor -- drives
    # the scaling: reference root-finding -> H^3, scan assembly+scan -> H^2.
    hi = H_arr >= 6
    Hh = H_arr[hi]
    i8 = A["H"].index(8)
    ax.loglog(Hh, eig_pf[i8] * (Hh / 8.0) ** 3, "--", color=RED, lw=1.0,
              alpha=0.55, label=r"$\propto H^3$ (asymptote)")
    ax.loglog(Hh, scan_pf[i8] * (Hh / 8.0) ** 2, "--", color=BLUE, lw=1.0,
              alpha=0.55, label=r"$\propto H^2$ (asymptote)")
    ax.set_xlabel("number of harmonics $H$")
    ax.set_ylabel("wall time per trial frequency [ms]")
    ax.set_title(f"single band, $N_{{\\rm obs}}={A['N']}$, $N_f={A['Nf']}$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"../plots/timing_vs_nharm.{ext}", dpi=150)
    plt.close(fig)

    _ndata_fig(results["ndata_const_cadence"],
               "constant cadence ($N_f = 12\\,N_{\\rm obs}$), $H=6$",
               "timing_vs_ndata",
               r"$\propto N_f N_{\rm obs}$ (extrapolated)")
    _ndata_fig(results["ndata_const_freq"],
               "constant baseline ($N_f=2000$), $H=6$",
               "timing_vs_ndata_const_freq",
               r"$\propto N_{\rm obs}$ (extrapolated)")


def _ndata_fig(block, title, fname, slow_scaling_label):
    N = np.array(block["N"], float)
    scan = np.array(block["scan"], float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.loglog(N, scan, "s-", color=BLUE, label="FTP scan (default)")
    eig = np.array([np.nan if v is None else v for v in block["eigvals"]])
    m = np.isfinite(eig)
    ax.loglog(N[m], eig[m], "o-", color=RED, label="FTP eigvals (reference)")
    slow = np.array([np.nan if v is None else v for v in block["slow"]])
    ms = np.isfinite(slow)
    ax.loglog(N[ms], slow[ms], "^-", color=GREY, label="non-linear optimisation")
    # extrapolate the non-linear O(N_f N_obs) trend as a dashed guide
    if ms.sum() >= 2:
        Nf_of = np.array(block["Nf"], float)
        Nlast = N[ms][-1]
        tlast = slow[ms][-1]
        Nf_last = Nf_of[N == Nlast][0]
        sel = N >= Nlast
        Next, Nf_ext = N[sel], Nf_of[sel]
        guide = tlast * (Next * Nf_ext) / (Nlast * Nf_last)
        ax.loglog(Next, guide, ":", color=GREY, lw=1.2, alpha=0.8,
                  label=slow_scaling_label)
    ax.set_xlabel("number of observations $N_{\\rm obs}$")
    ax.set_ylabel("wall time [s]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"../plots/{fname}.{ext}", dpi=150)
    plt.close(fig)


# ===========================================================================
def print_summary(results):
    A, D, E = results["vs_H"], results["H8_vs_Nf"], results["multiband_shared_phase"]
    print("\n" + "=" * 62)
    print("SUMMARY (measured this session; put ONLY these in the TeX)")
    print("=" * 62)

    sp = [e / s for e, s in zip(A["eigvals"], A["scan"])]
    print("Single-band scan-vs-eigvals speedup vs H (N=200, Nf=6000):")
    for h, s in zip(A["H"], sp):
        print(f"    H={h:2d}: {s:5.1f}x")
    print(f"    -> range over H=1..12: {min(sp):.1f}x -- {max(sp):.1f}x")

    print("Single-band scan-vs-eigvals at H=8 vs grid size:")
    for nf, e, s in zip(D["Nf"], D["eigvals"], D["scan"]):
        print(f"    Nf={nf:6d}: {e/s:5.1f}x   (eigvals {e:.2f}s -> scan {s:.3f}s)")

    print("Multiband shared_phase scan-vs-eigvals (K=2, dense grid):")
    for h, e, s in zip(E["H"], E["eigvals"], E["scan"]):
        print(f"    H={h}: {e/s:5.1f}x   (eigvals {e:.2f}s -> scan {s:.3f}s)")

    for key, tag in (("ndata_const_cadence", "const cadence"),
                     ("ndata_const_freq", "const baseline Nf=2000")):
        blk = results[key]
        print(f"FTP(scan) vs non-linear optimisation [{tag}] (measured points):")
        for N, sl, sc in zip(blk["N"], blk["slow"], blk["scan"]):
            if sl is not None:
                print(f"    N={N:4d}: {sl / sc:8.1f}x")
    print("=" * 62)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        with open("timing_results.json") as fh:
            results = json.load(fh)
    else:
        results = measure()
    make_figures(results)
    print_summary(results)
