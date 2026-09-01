"""Matplotlib tearsheet for a cross-sectional momentum run.

Color rules: the strategy is always blue, the benchmark is always gray
context, and the blue/orange pair is reserved for diverging polarity
(positive vs negative IC). One y-axis per panel, no exceptions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult
from .metrics import drawdown_series, summary

STRATEGY = "#2a78d6"   # categorical slot 1
NEGATIVE = "#eb6834"   # diverging warm pole, only for polarity
BENCHMARK = "#8a8984"  # muted context
GRID = "#d9d8d2"
INK = "#3d3d3a"

plt.rcParams.update(
    {
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def make_tearsheet(
    result: BacktestResult,
    ic: pd.Series,
    quantile_rets: pd.Series,
    benchmark_equity: pd.Series | None = None,
    out_path: str = "tearsheet.png",
    title: str = "Cross-sectional momentum",
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(title, fontsize=14, color=INK)

    ax = axes[0, 0]
    ax.plot(result.equity, color=STRATEGY, lw=2, label="Strategy")
    if benchmark_equity is not None:
        ax.plot(
            benchmark_equity.reindex(result.equity.index),
            color=BENCHMARK, lw=2, label="Buy & hold SPY",
        )
    ax.set_yscale("log")
    ax.set_title("Equity (log)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    dd = drawdown_series(result.equity)
    ax.fill_between(dd.index, dd, 0, color=STRATEGY, alpha=0.35)
    ax.plot(dd, color=STRATEGY, lw=1)
    ax.set_title("Drawdown")

    ax = axes[1, 0]
    colors = [STRATEGY if v >= 0 else NEGATIVE for v in ic]
    ax.bar(ic.index, ic.values, width=22, color=colors)
    ax.axhline(0, color=GRID, lw=1)
    mean_ic = ic.mean()
    ax.axhline(mean_ic, color=INK, lw=1, ls="--")
    ax.set_title(f"Monthly IC (mean {mean_ic:.3f})")

    ax = axes[1, 1]
    ax.bar(quantile_rets.index.astype(str), quantile_rets.values, color=STRATEGY)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_title("Mean forward return by signal decile")
    ax.set_xlabel("losers → winners")

    ax = axes[2, 0]
    ax.plot(result.long_exposure, color=STRATEGY, lw=1.5, label="Long")
    ax.plot(-result.short_exposure, color=NEGATIVE, lw=1.5, label="Short")
    ax.set_title("Gross exposure by side")
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.axis("off")
    stats = summary(result.equity, result.returns, result.turnover)
    lines = [f"{k:<18} {v:>10.4f}" for k, v in stats.items()]
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, va="top", family="monospace", fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
