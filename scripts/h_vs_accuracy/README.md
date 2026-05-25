# H-vs-accuracy experiment (paper Section 2.4)

This directory contains the reproduction scripts for Table 1 and Figure 2
of `paper_v5`.  The experiment quantifies how many Fourier harmonics
`H` are needed to represent canonical fixed-shape periodic signals,
and how that choice affects empirical period recovery on synthetic
noisy light curves.

## Layout

| File | Purpose |
|---|---|
| `templates.py` | Analytic canonical templates (RRab, RRc, transit, detached EB, ellipsoidal+beaming).  Runnable as `python templates.py` to render a preview PNG. |
| `fidelity.py` | **Tier A** — computes the cumulative $L^2$ explained-variance $F(H)$ via FFT of each template sampled at 8192 phases.  Writes `results/fidelity.pkl`. |
| `recovery.py` | **Tier B** — generates synthetic noisy light curves and runs FTP at `H = min_H(F{=}0.95)` plus the LS-equivalent (FTP at H=1), reporting the period-recovery rate at peak-to-peak SNR ∈ {3, 7, 15}.  Writes `results/recovery.pkl` incrementally. |
| `make_table_and_figure.py` | Reads both pickles; writes `results/table_h_vs_accuracy.tex` (included from `paper_v5.tex` via `\input`) and `../plots/h_vs_accuracy.pdf`. |
| `results/` | Generated outputs (pickles + preview PNG + table tex). |

## Reproduction

From the project venv (`FastTemplatePeriodogram/.venv` after
`pip install -e .` of the FTP library):

```sh
cd paper/FastTemplatePeriodogramPaper/scripts/h_vs_accuracy
python templates.py             # optional: regenerates the preview PNG
python fidelity.py              # ~1 s; writes results/fidelity.pkl
python recovery.py              # ~30-40 min; writes results/recovery.pkl
python make_table_and_figure.py # ~1 s; writes ../../plots/h_vs_accuracy.pdf + table tex
```

`make_table_and_figure.py` is tolerant of a missing `recovery.pkl` —
it produces a Tier-A-only figure and table in that case, which is
useful for quick iteration on the prose.

## Configuration

All experiment parameters are module-level constants:

- `fidelity.N_PHASE = 8192`, `H_MAX = 50`, thresholds `(0.90, 0.95, 0.99, 0.999)`.
- `recovery.N_OBS = 100`, `T_OBS = 100.0`, `MIN_FREQ = 0.01`, `MAX_FREQ = 5.0`,
  `SAMPLES_PER_PEAK = 10`, `REL_TOL = 5e-3`, `N_TRIALS = 30`,
  `SNR_LEVELS = (3.0, 7.0, 15.0)`, `RNG_SEED = 20260525`.

The frequency search range and grid density are set by explicit
physical bounds; **Nyquist is not used** (it does not apply to
irregularly sampled time series).

## Reviewer cross-reference

This experiment addresses Kipping K.12, K.13, K.17, K.19, K.20.  See
`paper/response_to_reviewers.md` for the per-comment responses and
`paper/revision_punch_list.md` for the status table.
