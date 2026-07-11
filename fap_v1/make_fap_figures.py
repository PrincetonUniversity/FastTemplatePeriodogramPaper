#!/usr/bin/env python
"""WP D2 -- paper-quality significance/FAP figures from the WP D1 null run.

Fresh, self-contained plotting script (never touches scripts/).  It reads ONLY
the committed Monte-Carlo artifact ``fap_v1/null_maxpower.npz`` (copied from
FastTemplatePeriodogram/experiments/fap/output_null/) and regenerates

  plots/fap_vs_threshold.pdf   -- empirical FAP vs max-power threshold for the
                                  FTP null at H in {1,3,6,12} (H=1 == GLS),
                                  with GEV fits overlaid as dashed curves;
  plots/nullmax_vs_K.pdf       -- the catalog-max statistic's null vs the
                                  number of vocabulary templates K in {1,2,4,8}
                                  (nested prefixes of the order-8 PAM vocab).

Fully deterministic: everything is a function of the npz contents (the GEV fit
is a deterministic MLE), so no seed is needed here.  Every number printed by
this script is recomputed from the npz, never quoted.

Usage:  python fap_v1/make_fap_figures.py   (from the paper repo root)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'null_maxpower.npz')
PLOTDIR = os.path.join(HERE, os.pardir, 'plots')

# Fixed-order categorical palette, CVD-validated (dataviz six-checks: all PASS
# on light surface; worst adjacent deutan dE 19 with dashed/solid + legend as
# secondary encoding).
PALETTE = ['#3366CC', '#EE6677', '#228833', '#AA3377']

FAP_GUIDE = 0.01


def survival(x):
    """Empirical survival function P(X > z) on the sorted sample points."""
    xs = np.sort(x)
    n = len(xs)
    # P(X > xs[i]) with the standard (n - i - 1 + 1)/n = 1 - i/n convention at
    # the left edge of each step; plot as steps-post.
    sf = 1.0 - np.arange(1, n + 1) / n
    return xs, sf


def gev_fit(x):
    """Deterministic scipy GEV MLE fit; returns (c, loc, scale)."""
    return stats.genextreme.fit(np.asarray(x, dtype=float))


def style_ax(ax):
    ax.grid(True, which='major', color='0.88', lw=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction='out', length=3)


def fap_vs_threshold(d, n_real):
    keys = [('ftp_H1', r'$H=1$ ($\equiv$ GLS)'),
            ('ftp_H3', r'$H=3$'),
            ('ftp_H6', r'$H=6$'),
            ('ftp_H12', r'$H=12$')]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    thr_grid = np.linspace(0.30, 0.68, 400)
    for (key, label), col in zip(keys, PALETTE):
        x = d[key]
        xs, sf = survival(x)
        m = sf > 0
        ax.plot(xs[m], sf[m], color=col, lw=1.6, drawstyle='steps-post',
                label=label, zorder=3)
        c, loc, scale = gev_fit(x)
        gev_sf = stats.genextreme.sf(thr_grid, c, loc, scale)
        mg = gev_sf > 0.5 / n_real
        ax.plot(thr_grid[mg], gev_sf[mg], color=col, lw=1.0, ls='--',
                alpha=0.85, zorder=2)
    ax.axhline(FAP_GUIDE, color='0.45', lw=0.8, ls=':', zorder=1)
    ax.text(0.305, FAP_GUIDE * 1.18, 'FAP = 0.01', color='0.35', fontsize=7,
            va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(0.8 / n_real, 1.05)
    ax.set_xlim(0.30, 0.68)
    ax.set_xlabel(r'max-power threshold $z$')
    ax.set_ylabel(r'false-alarm probability $\Pr(\max_\omega P > z)$')
    style_ax(ax)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=8,
                    title='FTP null (solid: empirical; dashed: GEV)',
                    title_fontsize=7.5)
    leg._legend_box.align = 'left'
    fig.tight_layout()
    out = os.path.join(PLOTDIR, 'fap_vs_threshold.pdf')
    fig.savefig(out)
    plt.close(fig)
    return out


def nullmax_vs_k(d, n_real):
    ks = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    thr01 = []
    for k, col in zip(ks, PALETTE):
        x = d['cat_K%d' % k]
        xs, sf = survival(x)
        m = sf > 0
        ax.plot(xs[m], sf[m], color=col, lw=1.6, drawstyle='steps-post',
                label=r'$K=%d$' % k, zorder=3)
        t = float(np.quantile(x, 1.0 - FAP_GUIDE))
        thr01.append(t)
        ax.plot([t], [FAP_GUIDE], marker='o', ms=4.5, color=col,
                mec='white', mew=0.8, zorder=4)
    ax.axhline(FAP_GUIDE, color='0.45', lw=0.8, ls=':', zorder=1)
    ax.text(0.305, FAP_GUIDE * 1.18, 'FAP = 0.01', color='0.35', fontsize=7,
            va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(0.8 / n_real, 1.05)
    ax.set_xlim(0.30, 0.68)
    ax.set_xlabel(r'catalog-max power threshold $z$')
    ax.set_ylabel(r'false-alarm probability $\Pr(\max_{j \leq K,\,\omega} P_j > z)$')
    style_ax(ax)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=8,
                    title='catalog-max null\n(nested $K$-template prefixes)',
                    title_fontsize=7.5)
    leg._legend_box.align = 'left'
    fig.tight_layout()
    out = os.path.join(PLOTDIR, 'nullmax_vs_K.pdf')
    fig.savefig(out)
    plt.close(fig)
    return out, thr01


def main():
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42,
    })
    d = np.load(NPZ, allow_pickle=True)
    cfg = json.loads(str(d['config_json']))
    n_real = int(cfg['n_real'])
    assert len(d['ftp_H1']) == n_real

    out1 = fap_vs_threshold(d, n_real)
    out2, thr01 = nullmax_vs_k(d, n_real)

    # Console report of every number the subsection quotes (recomputed here).
    print('n_real=%d n_obs=%d n_freq=%d band=single' %
          (n_real, cfg['n_obs'], cfg['n_freq']))
    print('max|FTP_H1 - GLS| = %.3e' % np.max(np.abs(d['ftp_H1'] - d['gls'])))
    for key in ('ftp_H1', 'ftp_H3', 'ftp_H6', 'ftp_H12'):
        x = d[key]
        print('%-8s mean=%.4f median=%.4f p99=%.4f thr(FAP=0.01)=%.3f'
              % (key, x.mean(), np.median(x), np.percentile(x, 99),
                 np.quantile(x, 0.99)))
    for k, t in zip((1, 2, 4, 8), thr01):
        x = d['cat_K%d' % k]
        print('cat_K%-2d  mean=%.4f median=%.4f thr(FAP=0.01)=%.3f'
              % (k, x.mean(), np.median(x), t))
    vpt = d['vocab_per_template']
    means = vpt.mean(axis=0)
    print('between-template spread of per-template null-max means = %.4f'
          % (means.max() - means.min()))
    print('wrote %s and %s' % (out1, out2))


if __name__ == '__main__':
    main()
