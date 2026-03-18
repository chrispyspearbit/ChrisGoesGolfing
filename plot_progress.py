#!/usr/bin/env python3
"""Generate progress.png step graph from results.tsv."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

def main():
    tsv_path = "results.tsv"
    out_path = "progress.png"

    df = pd.read_csv(tsv_path, sep="\t")
    df["experiment"] = range(1, len(df) + 1)

    kept = df[df["status"] == "keep"].copy()
    discarded = df[df["status"] == "discard"].copy()
    crashed = df[df["status"] == "crash"].copy()

    # Running best (cumulative min of kept val_bpb_quant)
    kept["running_best"] = kept["val_bpb_quant"].cummin()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot discarded/crashed as gray dots
    if len(crashed) > 0:
        ax.scatter(crashed["experiment"], crashed["val_bpb_quant"],
                   color="#999999", marker="x", s=60, zorder=2, label="crash")
    if len(discarded) > 0:
        ax.scatter(discarded["experiment"], discarded["val_bpb_quant"],
                   color="#cccccc", edgecolors="#999999", s=50, zorder=2, label="discard")

    # Plot kept as green dots
    if len(kept) > 0:
        ax.scatter(kept["experiment"], kept["val_bpb_quant"],
                   color="#2ecc71", edgecolors="black", s=70, zorder=3, label="keep")

        # Step line for running best
        ax.step(kept["experiment"], kept["running_best"],
                where="post", color="#27ae60", linewidth=2, zorder=2, label="running best")

        # Annotate kept experiments
        for _, row in kept.iterrows():
            desc = row["description"]
            if len(desc) > 30:
                desc = desc[:28] + "…"
            ax.annotate(desc, (row["experiment"], row["val_bpb_quant"]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=7, rotation=25, ha="left", va="bottom",
                        color="#2c3e50")

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("val_bpb_quant (lower is better)", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    n_total = len(df)
    n_kept = len(kept)
    best = kept["val_bpb_quant"].min() if len(kept) > 0 else float("nan")
    ax.set_title(f"Autoresearch Progress: {n_total} experiments, {n_kept} kept — best: {best:.6f}",
                 fontsize=13, fontweight="bold")

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
