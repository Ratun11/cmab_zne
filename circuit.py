#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw folded circuits with proper superscripts and tight vertical spacing,
independent of the QuTiP circuit drawer.

Outputs:
 - fold_lambda1.pdf
 - fold_lambda3_wrapped2.pdf
 - fold_lambda5_wrapped4.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------- Gate sequences WITH mathtext-friendly labels ----------------
def U_seq():
    # X^{-1/2}, Y^{+1/2}, Y^{+1/2}, X^{+1/2}, Z
    return [
        ("RX", -np.pi/2,  r"$X^{-1/2}$"),
        ("RY", +np.pi/2,  r"$Y^{1/2}$"),
        ("RY", +np.pi/2,  r"$Y^{1/2}$"),
        ("RX", +np.pi/2,  r"$X^{1/2}$"),
        ("Z",  None,      r"$Z$"),
    ]

def Ud_seq():
    # U†: Z, RX(-π/2), RY(-π/2), RY(-π/2), RX(+π/2)
    return [
        ("Z",  None,      r"$Z$"),
        ("RX", -np.pi/2,  r"$X^{-1/2}$"),
        ("RY", -np.pi/2,  r"$Y^{-1/2}$"),
        ("RY", -np.pi/2,  r"$Y^{-1/2}$"),
        ("RX", +np.pi/2,  r"$X^{1/2}$"),
    ]

def chunks_lambda1():
    return [U_seq()]  # 1 row

def chunks_lambda3_two_lines():
    # Row 1: U ; Row 2: U† U
    return [U_seq(), Ud_seq() + U_seq()]

def chunks_lambda5_four_lines():
    # Four rows with local Z insertions (balanced visually)
    line1 = U_seq()
    line2 = [("Z", None, r"$Z$")] + Ud_seq()
    line3 = [("Z", None, r"$Z$")] + U_seq()
    line4 = [("Z", None, r"$Z$")] + Ud_seq() + [("Z", None, r"$Z$")] + U_seq()
    return [line1, line2, line3, line4]

# ---------------- Minimal, tight custom renderer ----------------
from matplotlib.patches import FancyBboxPatch

def draw_row(ax, ops, gate_w=1.0, gap=0.45, y=0.0, box_h=0.6, fs=13):
    """
    Draw a single wire with gates.
    Ensures the wire is BEHIND the gate boxes via zorder.
    """
    n = len(ops)
    xs = np.arange(n) * (gate_w + gap)

    # Wire behind everything
    x0, x1 = xs.min() - 0.7, xs.max() + gate_w + 0.7
    ax.plot([x0, x1], [y, y], color="black", lw=1.2, zorder=0)

    # Qubit label
    ax.text(x0 - 0.8, y - 0.02, r"$q_0$", fontsize=12, va="center", zorder=5)

    # Gates above the wire
    for x, (name, _theta, label) in zip(xs, ops):
        color = "#66c2ff" if name in ("RX", "RY") else "#c84ee6"
        rect = FancyBboxPatch(
            (x, y - box_h/2), gate_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0, facecolor=color, alpha=0.95,
            zorder=3, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(x + gate_w/2, y, label, fontsize=fs,
                ha="center", va="center", color="white", zorder=4)

    ax.set_xlim(x0 - 0.2, x1 + 0.2)
    ax.set_ylim(y - 0.9, y + 0.9)
    ax.axis("off")


def draw_stacked(chunks, out_pdf, per_row_height=0.7, hspace=0.005, fs=13):
    """
    Draw multiple rows (chunks) tightly stacked and save to PDF.
    """
    rows = len(chunks)
    fig_w = 9
    fig_h = max(0.6, per_row_height * rows)
    fig, axes = plt.subplots(rows, 1, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = [axes]

    for ax, ops in zip(axes, chunks):
        draw_row(ax, ops, gate_w=1.05, gap=0.45, y=0.0, box_h=0.65, fs=fs)

    plt.subplots_adjust(hspace=hspace, top=0.98, bottom=0.06, left=0.06, right=0.995)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    # λ=1: one row
    draw_stacked(chunks_lambda1(), "fold_lambda1.pdf",
                 per_row_height=0.7, hspace=0.005, fs=14)

    # λ=3: two rows, very tight spacing
    draw_stacked(chunks_lambda3_two_lines(), "fold_lambda3_wrapped2.pdf",
                 per_row_height=0.7, hspace=0.005, fs=14)

    # λ=5: four rows, very tight spacing
    draw_stacked(chunks_lambda5_four_lines(), "fold_lambda5_wrapped4.pdf",
                 per_row_height=0.7, hspace=0.005, fs=13)

    print("Saved PDFs: fold_lambda1.pdf, fold_lambda3_wrapped2.pdf, fold_lambda5_wrapped4.pdf")

if __name__ == "__main__":
    main()
