# `scripts/figs_v5/` — fresh figure pipeline for paper_v5

This directory contains the v5 figure-generation code for paper Figs 1–3.
It is a deliberate ground-up rewrite of the 2017-era
`paper/FastTemplatePeriodogramPaper/scripts/generate_plots.py`, which is
off-limits (see `~/.claude/projects/.../memory/feedback_paper_scripts.md`
for the rule).

The only legacy artifact reused is the EB-template Fourier-coefficient pickle
at `../saved_results/eb_template_HAT-059-0780895.pkl` (the 10-harmonic
truncated-Fourier representation of the HATNet eclipsing binary
2MASS\,J02240150+5717241, also known as BD+56°603).

## Files

| File | Output | Purpose |
|---|---|---|
| `common.py` | — | Shared style, template loader, simulator, periodogram helpers |
| `fig1_templates_and_periodograms.py` | `../../plots/templates_and_periodograms.pdf` | Fig 1: H = 1, 2, 5, 9, 10 phase-folded fits + periodograms with MHLS / BLS overlays |
| `fig2_corrwgats.py`  | `../../plots/correlation_with_nonlinopt.pdf`  | Fig 2: FTP@H=10 peak vs slow non-linear-opt peak across MC realizations |
| `fig3_corrwhighh.py` | `../../plots/correlation_with_large_H.pdf`    | Fig 3: FTP@H=10 peak vs FTP@H=h peak for h ∈ {1, 2, 5, 9} |

## Running

Invoke from the paper repo root (`paper/FastTemplatePeriodogramPaper/`)
using the project venv:

```bash
PY=../../FastTemplatePeriodogram/.venv/bin/python

$PY scripts/figs_v5/fig1_templates_and_periodograms.py
$PY scripts/figs_v5/fig2_corrwgats.py --n-realizations 300
$PY scripts/figs_v5/fig3_corrwhighh.py --n-realizations 300
```

Fig 1 is a single deterministic realization (seed `20260525`), runs in
seconds. Figs 2 and 3 run MC ensembles in parallel via `multiprocessing`;
default 300 realizations takes roughly a minute or two on a recent
laptop.

## Reviewer comments addressed

- **L.6 (Lintott)** — Fig 1's missing-glyph "−" on `d⁻¹`. Fix in
  `common.setup_mpl()`: `axes.unicode_minus=False`, force a DejaVu Sans /
  Helvetica sans family (which ships a real minus), and embed the actual
  font glyphs in the PDF via `pdf.fonttype=42`. Verified by inspecting the
  output PDF that the minus renders as ASCII `-` (not a box).
- **JH.28 (Hartman)** — define R in captions. Done in the v5 tex (commit
  `70b3f22`) and re-confirmed here: R is the Pearson linear-correlation
  coefficient between the two peak-value arrays.
- **JH.31** — "an example of which is shown in Figure 2" wording. Done in
  the v5 tex.
- **JH.32** — "Similar to Figure 3" rewording of fig:corrwhighh caption.
  Done in the v5 tex.

## JH.29 empirical finding (P_slow vs P_FTP)

The reviewer asked "why is `P_gatspy` sometimes > `P_FTP` if they are
equivalent except for the possibility that gatspy picks up a local χ²
minimum?". In v5 we relabel the comparison `P_slow` (the
`SlowTemplatePeriodogram` brute-force non-linear-opt class in the FTP
codebase), not P_gatspy.

Result of the rerun (100 MC realizations, EB template, 80 observations,
σ=0.4, amplitude=1.0, frequency drawn uniformly from [1, 8] d⁻¹,
samples-per-peak=5, search range [0.5, 15] d⁻¹):

- **Pearson R = 0.992.**
- **0/100 realizations have P_slow > P_FTP.**
- FTP is always at least as accurate as the slow fitter; the slow
  fitter is occasionally trapped in a local V-minimum and reports a
  *lower* P, producing points below the diagonal at low P_FTP.

The v5 caption's framing ("in realizations where the slow fitter is
trapped in a local V-minimum, it reports a higher residual V and
therefore a lower peak P_slow than the FTP, producing the points below
the diagonal at low P_FTP") is therefore exactly what the data show.
The original JH.29 concern was driven by the old "P_gatspy" figure
that, on inspection, was making a different comparison; with the slow
non-linear-opt fitter inside the FTP library, FTP dominates everywhere.

## JVP.7 empirical finding (true-frequency recovery)

To answer JVP's sticky note on Figs 3, 4 ("how do we know each algorithm
is performing well? Should we compare against the true period?"), Figs
2 and 3 now color-code each scatter point by whether *both* algorithms
recovered the injected frequency to within `|df/f_true| < 0.01` (about
two grid bins at the default simulator settings). Filled black = both
recovered; hollow red = at least one missed. The "both recover" count
is also annotated in-panel.

Result of the rerun (300 MC realizations each):

- **Fig 2** (FTP@H=10 vs `SlowTemplatePeriodogram`, 300 realizations):
  R = 0.990; FTP recovers 293/300, slow 289/300, both 289/300.
- **Fig 3** (FTP@H=10 vs FTP@H=h for h ∈ {1,2,5,9}):

  | h | both recover (out of 300) |
  |---|---|
  | 1 | 0   |
  | 2 | 286 |
  | 5 | 290 |
  | 9 | 293 |

  H=10 reference alone recovers 293/300.

The Fig 3 H=1 panel is therefore visually striking: every scatter point
is a hollow red circle even though Pearson R ≈ 0.6. This is exactly the
failure mode JVP was worried about (high correlation does not imply
correct recovery), and it disappears entirely by H ≥ 2 (where the
template is rich enough to track the EB shape).

## JH.30 empirical finding (low-H vs high-H P_FTP)

The reviewer asked "why is the lower harmonic P_FTP sometimes greater
than the higher harmonic P_FTP? Is this due to noise in the simulated
light curve?". Yes.

Result of the rerun (200 MC realizations, same simulator settings as
above, all five H values evaluated on the same data and frequency
grid):

| h vs H=10 | Pearson R | P_h > P_10 | "meaningful" (> 0.001 above) |
|---|---|---|---|
| h = 1 | 0.613 | 32/200 (16%) | 32 |
| h = 2 | 0.834 | 28/200 (14%) | 27 |
| h = 5 | 0.972 | 70/200 (35%) | 66 |
| h = 9 | 1.000 | 102/200 (51%) | 56 |

The pattern is consistent with the noise-overfit explanation:

- At very low H (1, 2) the model is underspecified for the EB shape, so
  it fits a significantly smaller fraction of the signal variance; it
  exceeds H=10 only in a minority of realizations, and the excess is
  small.
- At h=5 the model captures most of the signal but the H=10 model has
  five extra degrees of freedom that, at the chosen SNR, are largely
  fitting noise. This explains why h=5 exceeds h=10 in 35% of cases:
  the noise-fitting penalty cancels and slightly outweighs the
  small remaining signal residual.
- At h=9, h=10 and h=9 differ only by one Fourier component (the
  10th harmonic). R = 1.000 to three figures, and the "exceeds" rate is
  essentially 50% (51%), as expected for two near-identical estimators
  on noisy data.

The body text quantification of "up to 6H−1 local V minima" (paper_v5
§3.1.1, JH.27) does not depend on this — that's a per-frequency
algebraic statement, while JH.30 is about the maximum across the
frequency grid.

## Reproducibility

Both `fig2_corrwgats.py` and `fig3_corrwhighh.py` use a top-level seed
of `20260525` and derive per-realization sub-seeds via
`np.random.default_rng().integers(...)`. The same `--n-realizations`
gives bit-identical results across runs as long as Python / numpy
versions don't change.

`fig1_templates_and_periodograms.py` uses the same top-level seed for
its single realization.

## Constraints respected

- No code under `scripts/` outside this directory is imported, executed,
  or read.
- No modification to `ftperiodogram/` itself.
- `paper_v5.tex` and the existing PDFs in `plots/` are overwritten *in
  place* (the three figures named above); no caption changes were made
  here — the captions were finalized in tex commit `70b3f22`.
