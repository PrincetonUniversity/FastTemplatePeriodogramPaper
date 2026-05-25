"""Generate the H-vs-accuracy figure and LaTeX table for paper_v5.

Outputs (paths relative to the paper repo root):

  plots/h_vs_accuracy.pdf
      Two-panel figure.  Panel (a): cumulative L^2 explained-variance
      F(H) versus H for each canonical template, with horizontal
      threshold guides at F = 0.90, 0.95, 0.99.  Panel (b): empirical
      period-recovery rate versus SNR for FTP@H (template-shape
      truncation at the 95%-fidelity H) and the LS-equivalent FTP@1.

  scripts/h_vs_accuracy/results/table_h_vs_accuracy.tex
      LaTeX ``tabular`` content for Table 1, including the H-min
      columns and a recovery-rate column at SNR = 7.

Reads fidelity.pkl unconditionally; recovery.pkl is optional (figure
omits panel (b) and table omits recovery columns when missing).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from templates import TEMPLATES, PRETTY


HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS_DIR = HERE.parent.parent / "plots"

THRESHOLDS = (0.90, 0.95, 0.99)
SNR_FOR_TABLE = 7.0

# Stable, distinguishable styles per template.  Transit and EB share a
# closely-overlapping fidelity curve (the secondary contributes little
# energy); the dash style on EB lets the reader resolve them in panel (a).
STYLES = {
    "rrab":               ("C0", "o", "-"),
    "rrc":                ("C1", "s", "-"),
    "transit":            ("C3", "^", "-"),
    "eb_detached":        ("C2", "D", "--"),
    "ellipsoidal_beaming":("C4", "v", "-"),
}


def _load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_min_h(value):
    """Render min_h_at[t]: int or 'None' -> '$>\!50$' (the H_MAX cap)."""
    return r"$>\!50$" if value is None else f"{value}"


def _format_rate(value):
    return f"{int(round(value * 100))}\\%"


def write_table(fidelity, recovery=None):
    rows = []
    for key in TEMPLATES:
        cells = [PRETTY[key]]
        cells.extend(_format_min_h(fidelity[key]["min_h_at"][t])
                     for t in THRESHOLDS)
        if recovery is not None and key in recovery:
            rec = recovery[key]["by_snr"][SNR_FOR_TABLE]
            cells.append(f"$H={recovery[key]['h_used']}$")
            cells.append(_format_rate(rec["ftp_rate"]))
            cells.append(_format_rate(rec["ls_rate"]))
        rows.append(" & ".join(cells) + r" \\")

    has_recovery = recovery is not None
    n_cols = 4 + (3 if has_recovery else 0)
    align = "l" + "c" * (n_cols - 1)

    header_top = (r"Template & \multicolumn{3}{c}{$H_{\min}(F)$}"
                  + (r" & \multicolumn{3}{c}{Recovery at SNR\,$=7$}"
                     if has_recovery else "")
                  + r" \\")
    header_bot = ("& " + " & ".join(f"{int(t*100)}\\%" for t in THRESHOLDS)
                  + (r" & $H$ & FTP & LS-eq." if has_recovery else "")
                  + r" \\")

    # Plain \cline{} works in standard LaTeX without booktabs.
    sub_rules = r"\cline{2-4}"
    if has_recovery:
        sub_rules += r" \cline{5-7}"

    tabular = "\n".join([
        r"\begin{tabular}{" + align + "}",
        r"\hline",
        header_top,
        sub_rules,
        header_bot,
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ])

    out = RESULTS / "table_h_vs_accuracy.tex"
    out.write_text(tabular + "\n")
    print(f"wrote {out}")
    return out


def make_figure(fidelity, recovery=None):
    has_recovery = (recovery is not None
                    and all(k in recovery for k in TEMPLATES))
    if has_recovery:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig, ax_a = plt.subplots(1, 1, figsize=(6, 4.2))
        ax_b = None

    # --- Panel (a): cumulative fidelity vs H ---
    for key in TEMPLATES:
        color, marker, linestyle = STYLES[key]
        h = fidelity[key]["h"]
        cum = fidelity[key]["cum_fidelity"]
        ax_a.plot(h, cum, color=color, marker=marker, ls=linestyle,
                  markersize=4, markevery=2, lw=1.4,
                  label=PRETTY[key])
    for t in THRESHOLDS:
        ax_a.axhline(t, color="gray", lw=0.7, ls=":", alpha=0.7)
        ax_a.text(0.5, t + 0.005, f"{int(t*100)}%",
                  fontsize=7, color="gray", va="bottom")
    ax_a.set_xlabel(r"Number of harmonics $H$")
    ax_a.set_ylabel(r"Cumulative explained variance  $F(H)$")
    ax_a.set_xlim(left=0.5)
    ax_a.set_ylim(0.0, 1.005)
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8, loc="lower right")
    ax_a.set_title("(a)  $L^2$ fidelity of the H-truncated Fourier series",
                   fontsize=10)

    # --- Panel (b): recovery vs SNR ---
    if has_recovery:
        snrs = sorted(next(iter(recovery.values()))["by_snr"].keys())
        for key in TEMPLATES:
            color, marker, _ = STYLES[key]
            r_ftp = [recovery[key]["by_snr"][s]["ftp_rate"] for s in snrs]
            r_ls = [recovery[key]["by_snr"][s]["ls_rate"] for s in snrs]
            ax_b.plot(snrs, r_ftp, color=color, marker=marker,
                      ls="-", lw=1.4, label=PRETTY[key])
            ax_b.plot(snrs, r_ls, color=color, marker=marker,
                      ls=":", lw=1.4, alpha=0.6)

        # legend explainer for solid vs dotted
        ax_b.plot([], [], color="black", ls="-",
                  label=r"FTP at $H_{95\%}$")
        ax_b.plot([], [], color="black", ls=":", alpha=0.6,
                  label=r"LS-equivalent (FTP at $H{=}1$)")
        ax_b.set_xlabel("Peak-to-peak SNR")
        ax_b.set_ylabel("Period-recovery rate")
        ax_b.set_ylim(-0.03, 1.05)
        ax_b.grid(alpha=0.3)
        ax_b.legend(fontsize=7, ncol=1, loc="lower right")
        ax_b.set_title("(b)  Period recovery on synthetic LCs",
                       fontsize=10)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "h_vs_accuracy.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    plt.close(fig)


def main():
    fidelity = _load("fidelity.pkl")
    if fidelity is None:
        raise SystemExit("missing results/fidelity.pkl — run fidelity.py first")
    recovery = _load("recovery.pkl")
    if recovery is None:
        print("note: results/recovery.pkl missing; producing Tier A only")

    write_table(fidelity, recovery)
    make_figure(fidelity, recovery)


if __name__ == "__main__":
    main()
