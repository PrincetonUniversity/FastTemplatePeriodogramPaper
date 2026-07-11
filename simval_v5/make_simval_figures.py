#!/usr/bin/env python
"""WP F2: recompute + render everything the vocabulary/sim-validation sections quote.

Fresh consumer of the committed WP B8 rerun artifacts (the canonical Phase 3.2
production run; supersedes ``headline_full/``):

    FastTemplatePeriodogram/experiments/phase3_recovery/rerun_202606/raw/<tag>/out/
        results.json                 (per-seed curves + config + grid provenance)
        per_source/seed*__*.npz      (WP B1: per-source P_true/P_rec/recovered/baseline)

Everything quoted in paper_v5.tex Sections "Template vocabularies ..." and
"Validation on simulated multi-band surveys" (and the alias-breakdown appendix
table) is recomputed here from those artifacts -- pooled Wilson 95% CIs from
counts, exact McNemar paired contrasts from the persisted per-source masks,
the SE-aware knee, the 10k-vs-20k grid-convergence test, the empirical/synthetic
error-curve ratio, the cost panel, and the phase-coherence alias re-scoring --
then written to ``derived_numbers.json`` (the TeX numbers' single source) and
rendered to ``../plots/recovery_vs_nepochs_sesar.pdf`` and
``../plots/recovery_vs_k_sparse.pdf``.

Run with the code repo's venv (needs ftperiodogram + numpy + matplotlib):

    ../../FastTemplatePeriodogram/.venv/bin/python make_simval_figures.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_REPO = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'FastTemplatePeriodogram'))
RERUN = os.path.join(CODE_REPO, 'experiments', 'phase3_recovery', 'rerun_202606')
RAW = os.path.join(RERUN, 'raw')
PLOTS = os.path.abspath(os.path.join(HERE, '..', 'plots'))

sys.path.insert(0, CODE_REPO)
from ftperiodogram.validation import mcnemar_test, wilson_interval   # noqa: E402
from ftperiodogram.recovery import rescore_aliases                   # noqa: E402
from ftperiodogram.simulate import exp_mag_error                     # noqa: E402

Z95 = 1.959963984540054
METHODS = ('ftp_pam', 'ftp_greedy', 'gls', 'mhls', 'mbls', 'ce')
SESAR_TAGS = ('a-sesar-0', 'a-sesar-1', 'a-sesar-2')


def load(tag):
    with open(os.path.join(RAW, tag, 'out', 'results.json')) as fh:
        return json.load(fh)


def npz(tag, seed, method, cell):
    return np.load(os.path.join(RAW, tag, 'out', 'per_source',
                                'seed%d__%s__%s.npz' % (seed, method, cell)))


def pooled_wilson(per_seed_rates, n_per_seed):
    """Pool exact per-seed rates back to counts; Wilson 95% CI on the pool."""
    a = np.atleast_2d(np.asarray(per_seed_rates, dtype=float))
    n = np.asarray(n_per_seed, dtype=float)
    k = np.rint(a * n[:, None]).sum(axis=0)
    n_tot = float(n.sum())
    lo, hi = wilson_interval(k, n_tot)
    return {'mean': (k / n_tot).tolist(), 'lo': np.atleast_1d(lo).tolist(),
            'hi': np.atleast_1d(hi).tolist(), 'n_pooled': int(n_tot)}


def wilson_se(rate, n):
    st = pooled_wilson([[rate]], [n])
    return (st['hi'][0] - st['lo'][0]) / (2.0 * Z95)


def se_aware_knee(k_values, rates, n_pooled):
    """Smallest K whose pooled Wilson interval overlaps the argmax-K interval."""
    iv = [pooled_wilson([[r]], [n_pooled]) for r in rates]
    best = int(np.argmax(rates))
    for i, k in enumerate(k_values):
        if iv[i]['hi'][0] >= iv[best]['lo'][0]:
            return int(k)
    return int(k_values[best])


def knee_frac(k_values, rates, frac=0.98):
    rmax = max(rates)
    for k, r in zip(k_values, rates):
        if r >= frac * rmax:
            return int(k)
    return int(k_values[-1])


def pooled_mask(tags_seeds, method, cell):
    return np.concatenate([np.asarray(npz(t, s, method, cell)['recovered'],
                                      dtype=bool) for t, s in tags_seeds])


def main():
    out = {}
    sesar = [load(t) for t in SESAR_TAGS]
    seeds = [r['per_seed'][0] for r in sesar]
    cfg = sesar[0]['config']
    n_per_seed = [s['n_epochs_sweep']['n_sources'] for s in seeds]
    n_pooled = int(sum(n_per_seed))
    ts = list(zip(SESAR_TAGS, [s['seed'] for s in seeds]))

    # grid provenance (WP B7): df and points per Rayleigh width, recomputed
    df = (cfg['f_max'] - cfg['f_min']) / (cfg['n_freq'] - 1)
    rayleigh = 1.0 / cfg['baseline_days']
    out['config'] = {
        'H': cfg['nharmonics'], 'obs_bands': cfg['obs_bands'],
        'n_sources_per_seed': cfg['n_sources'], 'n_seeds': len(seeds),
        'n_pooled': n_pooled, 'f_min': cfg['f_min'], 'f_max': cfg['f_max'],
        'n_freq': cfg['n_freq'], 'baseline_days': cfg['baseline_days'],
        'grid_df': df, 'pts_per_rayleigh': rayleigh / df,
        'pts_per_h8_peak': rayleigh / df / cfg['nharmonics'],
        'dense_master_epochs': cfg['dense_master_epochs'],
        'sparse_master_epochs': cfg['sparse_master_epochs'],
        'mhls_h': cfg['mhls_h'], 'mbls_h': cfg['mbls_h'],
        'n_library_sesar': sesar[0]['n_universe'],
        'n_library_bv': load('a-bv-0')['n_universe']}
    assert abs(sesar[0]['grid_df'] - df) < 1e-12
    # results.json rounds pts/Rayleigh to 3 decimals; recompute must agree there
    assert abs(sesar[0]['grid_points_per_rayleigh'] - rayleigh / df) < 5e-4

    # ------------------------------------------------------------------
    # N-sweep, pooled sesar: Wilson CIs + pooled exact McNemar vs FTP(PAM)
    # ------------------------------------------------------------------
    nsw = seeds[0]['n_epochs_sweep']
    n_values = nsw['n_epochs_values']
    out['nsweep_sesar'] = {'n_epochs_values': n_values, 'k': nsw['k']}
    for m in METHODS:
        out['nsweep_sesar'][m] = pooled_wilson(
            [s['n_epochs_sweep'][m] for s in seeds], n_per_seed)
    mcn = {}
    for m in METHODS[2:] + ('ftp_greedy',):
        mcn[m] = []
        for N in n_values:
            a = pooled_mask(ts, 'ftp_pam', 'nsweep-N%d' % N)
            b = pooled_mask(ts, m, 'nsweep-N%d' % N)
            a_only, b_only, p = mcnemar_test(a, b)
            mcn[m].append({'N': N, 'ftp_only': a_only, 'other_only': b_only,
                           'p': p})
    out['nsweep_sesar']['mcnemar_vs_ftp_pam'] = mcn
    # MHLS cap sanity: at N=4/band x 2 bands the cap forces H_eff = 1 == GLS
    m4 = pooled_mask(ts, 'mhls', 'nsweep-N4') == pooled_mask(ts, 'gls', 'nsweep-N4')
    out['nsweep_sesar']['mhls_equals_gls_at_N4'] = bool(m4.all())

    # ------------------------------------------------------------------
    # Sparse K-sweep: pooled sesar + single-seed BV, knees, K1-vs-K2 McNemar
    # ------------------------------------------------------------------
    ks = seeds[0]['k_sweep_sparse']
    k_values = ks['k_values']
    pam = pooled_wilson([s['k_sweep_sparse']['ftp_pam'] for s in seeds],
                        n_per_seed)
    grd = pooled_wilson([s['k_sweep_sparse']['ftp_greedy'] for s in seeds],
                        n_per_seed)
    base = {b: pooled_wilson([[s['k_sweep_sparse']['baselines'][b]]
                              for s in seeds], n_per_seed)
            for b in ks['baselines']}
    a1 = pooled_mask(ts, 'ftp_pam', 'ksweep-sparse-K1')
    a2 = pooled_mask(ts, 'ftp_pam', 'ksweep-sparse-K2')
    k1o, k2o, pk = mcnemar_test(a1, a2)
    out['ksweep_sesar'] = {
        'k_values': k_values, 'n_epochs': ks['n_epochs'],
        'ftp_pam': pam, 'ftp_greedy': grd, 'baselines': base,
        'knee_frac98': knee_frac(k_values, pam['mean']),
        'knee_se_aware': se_aware_knee(k_values, pam['mean'], n_pooled),
        'mcnemar_K1_vs_K2': {'K1_only': k1o, 'K2_only': k2o, 'p': pk}}

    bv = load('a-bv-0')
    bvs = bv['per_seed'][0]
    nbv = [bvs['k_sweep_sparse']['n_sources']]
    bpam = pooled_wilson([bvs['k_sweep_sparse']['ftp_pam']], nbv)
    b1 = np.asarray(npz('a-bv-0', 0, 'ftp_pam', 'ksweep-sparse-K1')['recovered'],
                    dtype=bool)
    b2 = np.asarray(npz('a-bv-0', 0, 'ftp_pam', 'ksweep-sparse-K2')['recovered'],
                    dtype=bool)
    bk1o, bk2o, bpk = mcnemar_test(b1, b2)
    out['ksweep_bv'] = {
        'k_values': bvs['k_sweep_sparse']['k_values'],
        'n_epochs': bvs['k_sweep_sparse']['n_epochs'],
        'ftp_pam': bpam,
        'ftp_greedy': pooled_wilson([bvs['k_sweep_sparse']['ftp_greedy']], nbv),
        'baselines': {b: pooled_wilson([[bvs['k_sweep_sparse']['baselines'][b]]],
                                       nbv)
                      for b in bvs['k_sweep_sparse']['baselines']},
        'k1_to_k2_jump': bpam['mean'][1] - bpam['mean'][0],
        'knee_frac98': knee_frac(bvs['k_sweep_sparse']['k_values'],
                                 bpam['mean']),
        'knee_se_aware': se_aware_knee(bvs['k_sweep_sparse']['k_values'],
                                       bpam['mean'], nbv[0]),
        'mcnemar_K1_vs_K2': {'K1_only': bk1o, 'K2_only': bk2o, 'p': bpk},
        'nsweep': {m: bvs['n_epochs_sweep'][m] for m in METHODS},
        'nsweep_k': bvs['n_epochs_sweep']['k']}

    # ------------------------------------------------------------------
    # Robustness arms (single seed each): rates + FTP(PAM)>=GLS check
    # ------------------------------------------------------------------
    out['robustness'] = {}
    for tag in ('b-holdout-0', 'b-xuniv-0', 'b-bandamp-0'):
        s = load(tag)['per_seed'][0]
        nswr = s['n_epochs_sweep']
        arm = {m: nswr[m] for m in METHODS}
        arm['n_epochs_values'] = nswr['n_epochs_values']
        arm['k'] = nswr['k']
        arm['n_sources'] = nswr['n_sources']
        arm['arms'] = {k: v for k, v in s['arms'].items()
                       if not isinstance(v, list)}
        arm['ftp_pam_ge_gls_all_N'] = bool(all(
            f >= g for f, g in zip(nswr['ftp_pam'], nswr['gls'])))
        arm['wilson'] = {m: pooled_wilson([nswr[m]], [nswr['n_sources']])
                         for m in ('ftp_pam', 'ftp_greedy', 'gls')}
        out['robustness'][tag] = arm

    # ------------------------------------------------------------------
    # Grid convergence: paired exact McNemar, 20k vs 10k, identical seed-0 sources
    # ------------------------------------------------------------------
    c = load('c-grid20k-0')['per_seed'][0]['n_epochs_sweep']
    a0 = seeds[0]['n_epochs_sweep']
    conv = []
    for m in ('ftp_pam', 'gls', 'mhls', 'mbls', 'ce'):
        for i, N in enumerate(c['n_epochs_values']):
            za = np.asarray(npz('a-sesar-0', 0, m, 'nsweep-N%d' % N)['recovered'],
                            dtype=bool)
            zc = np.asarray(npz('c-grid20k-0', 0, m, 'nsweep-N%d' % N)['recovered'],
                            dtype=bool)
            only10, only20, p = mcnemar_test(za, zc)
            conv.append({'method': m, 'N': N, 'r20k': c[m][i], 'r10k': a0[m][i],
                         'only20': only20, 'only10': only10, 'p': p})
    sig = [r for r in conv if r['p'] < 0.05]
    gap = {N: {'g10': a0['ftp_pam'][a0['n_epochs_values'].index(N)]
               - a0['gls'][a0['n_epochs_values'].index(N)],
               'g20': c['ftp_pam'][c['n_epochs_values'].index(N)]
               - c['gls'][c['n_epochs_values'].index(N)]}
           for N in (4, 8, 12)}
    out['grid_convergence'] = {
        'n_freq_production': cfg['n_freq'], 'n_freq_check': 20000,
        'pts_per_rayleigh_20k': rayleigh / ((cfg['f_max'] - cfg['f_min'])
                                            / (20000 - 1)),
        'cells': conv, 'n_significant': len(sig), 'n_cells': len(conv),
        'all_significant_gain_at_20k': bool(all(r['r20k'] >= r['r10k']
                                                for r in sig)),
        'gls_mbls_all_flat': bool(all(r['p'] > 0.05 for r in conv
                                      if r['method'] in ('gls', 'mbls'))),
        'min_p_gls_mbls': min(r['p'] for r in conv
                              if r['method'] in ('gls', 'mbls')),
        'ftp_minus_gls_gap_10k_vs_20k': gap}

    # ------------------------------------------------------------------
    # Empirical-error arm (d) + error-curve ratio, recomputed from the
    # committed ztf_error_curve.json against the synthetic exp_mag_error
    # ------------------------------------------------------------------
    d = load('d-empirical-0')['per_seed'][0]
    dn = d['n_epochs_sweep']
    out['empirical'] = {
        'err_model': d['err_model'], 'fixed_k': d['fixed_k'],
        'n_epochs_values': dn['n_epochs_values'],
        'ftp_pam': dn['ftp_pam'], 'gls': dn['gls'],
        'ftp_pam_synthetic_pooled': out['nsweep_sesar']['ftp_pam']['mean'],
        'cost': d['cost']}
    with open(os.path.join(RERUN, os.pardir, 'ztf_error_curve.json')) as fh:
        curve = json.load(fh)
    mags = np.asarray(curve['mag_centers'], dtype=float)
    med = np.asarray(curve['sigma_medians'], dtype=float)
    synth = exp_mag_error()(mags)
    ratio = med / synth
    out['empirical']['error_ratio_range'] = [float(ratio.min()),
                                             float(ratio.max())]
    out['empirical']['error_ratio_mag_range'] = [float(mags.min()),
                                                 float(mags.max())]

    # ------------------------------------------------------------------
    # Alias breakdown (WP B5): phase-coherence re-scoring, pooled sesar seeds
    # ------------------------------------------------------------------
    rows = []
    for m in METHODS:
        for N in n_values:
            arrs = [npz(t, s, m, 'nsweep-N%d' % N) for t, s in ts]
            p_rec = np.concatenate([np.asarray(a['p_rec'], float) for a in arrs])
            p_true = np.concatenate([np.asarray(a['p_true'], float)
                                     for a in arrs])
            bl = np.concatenate([np.asarray(a['baseline'], float)
                                 for a in arrs])
            _, breakdown = rescore_aliases(p_rec, p_true, baseline=bl,
                                           criterion='phase_coherence',
                                           delta_phi_max=0.5)
            rows.append({'method': m, 'N': int(N), 'breakdown': breakdown})
    out['alias_breakdown'] = {'criterion': 'phase_coherence',
                              'delta_phi_max': 0.5, 'n_pooled': n_pooled,
                              'rows': rows}

    with open(os.path.join(HERE, 'derived_numbers.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', os.path.join(HERE, 'derived_numbers.json'))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {
        'ftp_pam': dict(color='tab:blue', marker='o', ls='-',
                        label='FTP (PAM vocabulary)'),
        'ftp_greedy': dict(color='tab:purple', marker='D', ls='-',
                           label='FTP (greedy vocabulary)'),
        'gls': dict(color='0.45', marker='s', ls='--', label='GLS'),
        'mhls': dict(color='tab:red', marker='v', ls=':',
                     label='MHLS (capped $H$)'),
        'mbls': dict(color='tab:green', marker='^', ls='-.',
                     label='multiband LS'),
        'ce': dict(color='tab:orange', marker='x', ls=(0, (3, 1, 1, 1)),
                   label='conditional entropy'),
    }

    def band(ax, x, st, key):
        kw = dict(style[key])
        line, = ax.plot(x, st['mean'], **kw)
        ax.fill_between(x, st['lo'], st['hi'], color=line.get_color(),
                        alpha=0.15, lw=0)

    # (i) recovery vs N_epochs, pooled sesar
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in METHODS:
        band(ax, n_values, out['nsweep_sesar'][m], m)
    ax.set_xlabel('epochs per band $N$')
    ax.set_ylabel('period recovery rate')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(n_values)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'recovery_vs_nepochs_sesar.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    # (ii) sparse recovery vs K: sesar (pooled) + BV (one seed)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, blk, title in (
            (axes[0], out['ksweep_sesar'],
             'RRab-dominated library (Sesar et al. 2010)'),
            (axes[1], out['ksweep_bv'],
             'RRab+RRc library (Baeza-Villagra et al. 2025)')):
        kv = blk['k_values']
        band(ax, kv, blk['ftp_pam'], 'ftp_pam')
        band(ax, kv, blk['ftp_greedy'], 'ftp_greedy')
        for name in ('gls', 'mhls', 'mbls'):
            b = blk['baselines'][name]
            ax.axhline(b['mean'][0], color=style[name]['color'],
                       ls=style[name]['ls'], lw=1.4, label=style[name]['label'])
            ax.axhspan(b['lo'][0], b['hi'][0], color=style[name]['color'],
                       alpha=0.08, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(kv)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('vocabulary size $K$')
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('period recovery rate')
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'recovery_vs_k_sparse.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print('wrote figures to', PLOTS)

    # console digest of the numbers the TeX quotes
    nsm = out['nsweep_sesar']
    print('pooled sesar N-sweep FTP(PAM):',
          ['%.3f' % v for v in nsm['ftp_pam']['mean']])
    print('  GLS:', ['%.3f' % v for v in nsm['gls']['mean']])
    print('  McNemar FTP vs GLS @N=8:', mcn['gls'][1])
    print('sesar sparse PAM K-curve:',
          ['%.3f' % v for v in out['ksweep_sesar']['ftp_pam']['mean']],
          'knee(frac)=%d se-aware=%d' % (out['ksweep_sesar']['knee_frac98'],
                                         out['ksweep_sesar']['knee_se_aware']),
          'K1-vs-K2 p=%.3g' % out['ksweep_sesar']['mcnemar_K1_vs_K2']['p'])
    print('bv sparse PAM K-curve:',
          ['%.3f' % v for v in out['ksweep_bv']['ftp_pam']['mean']],
          'jump=%.3f' % out['ksweep_bv']['k1_to_k2_jump'],
          'K1-vs-K2 p=%.3g' % out['ksweep_bv']['mcnemar_K1_vs_K2']['p'])
    print('grid: %d/%d cells significant; GLS/MBLS min p=%.3f; gaps %s'
          % (out['grid_convergence']['n_significant'],
             out['grid_convergence']['n_cells'],
             out['grid_convergence']['min_p_gls_mbls'],
             out['grid_convergence']['ftp_minus_gls_gap_10k_vs_20k']))
    print('empirical error ratio %.2f-%.2f over mags %.1f-%.1f'
          % (*out['empirical']['error_ratio_range'],
             *out['empirical']['error_ratio_mag_range']))
    print('cost:', out['empirical']['cost'])
    print('pts/Rayleigh = %.3f (10k), %.3f (20k); per H=8 peak width %.3f'
          % (out['config']['pts_per_rayleigh'],
             out['grid_convergence']['pts_per_rayleigh_20k'],
             out['config']['pts_per_h8_peak']))


if __name__ == '__main__':
    main()
