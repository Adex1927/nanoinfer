#!/usr/bin/env python3
"""
Plot the probe data written by nanoinfer (probe.csv).

CSV columns: name,phase,token_pos,layer,ms

Prints a text table and saves one figure with four plain panels:
  1. decode step latency vs token position   (trend / outliers)
  2. share of a decode step by stage         (what to optimize)
  3. mean time per layer, per stage          (are any layers odd?)
  4. attention time vs context length        (O(pos) growth of mha)

Usage:
  python3 tools/plot_probe.py                    # reads probe.csv -> probe.png
  python3 tools/plot_probe.py run.csv -o run.png
  python3 tools/plot_probe.py --show
"""
import argparse

import numpy as np
import pandas as pd

# Stages that make up one token's work. Aggregates (all_layers,
# decode:forward_token, prefill, prefill_total, decode:argmax) are excluded
# from the breakdown so nothing is double counted.
LAYER_STAGES = [
    "layer:attn_norm", "layer:qkv_proj", "layer:rope",
    "layer:mha", "layer:attn_out_proj", "layer:ffn",
]
TOKEN_STAGES = ["embedding_lookup", "final_norm_and_lm_head"]


def print_table(df):
    g = df.groupby(["name", "phase"]).ms
    t = g.agg(n="count", total="sum", mean="mean",
              p50="median", p99=lambda x: x.quantile(0.99), max="max")
    t = t.round(3)
    print(t.to_string())


def per_token_breakdown(dec):
    """ms per decode token, per stage (layer stages summed over layers)."""
    parts = {}
    for st in LAYER_STAGES + TOKEN_STAGES:
        d = dec[dec.name == st]
        parts[st] = d.groupby("token_pos").ms.sum()
    return pd.DataFrame(parts).fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="probe.csv")
    ap.add_argument("-o", "--out", default="probe.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    dec = df[df.phase == "decode"]

    print_table(df)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs.ravel()

    # 1. decode step latency vs token position
    step = dec[dec.name == "decode:forward_token"].sort_values("token_pos")
    if not step.empty:
        ax[0].plot(step.token_pos, step.ms, marker=".", lw=1)
        ax[0].axhline(step.ms.median(), color="gray", ls="--", lw=1,
                      label=f"median {step.ms.median():.0f} ms")
        ax[0].legend()
    ax[0].set_title("Decode step latency")
    ax[0].set_xlabel("token position")
    ax[0].set_ylabel("ms")
    ax[0].set_ylim(bottom=0)

    # 2. share of a decode step by stage
    bd = per_token_breakdown(dec)
    mean_ms = bd.mean().sort_values()
    share = 100 * mean_ms / mean_ms.sum()
    ax[1].barh(mean_ms.index, mean_ms.values)
    for i, (ms, pct) in enumerate(zip(mean_ms.values, share.values)):
        ax[1].text(ms, i, f" {ms:.1f} ms ({pct:.1f}%)", va="center", fontsize=8)
    ax[1].set_title("Mean ms per decode token, by stage")
    ax[1].set_xlabel("ms (layer stages summed over layers)")
    ax[1].set_xlim(right=mean_ms.max() * 1.35)

    # 3. mean time per layer, per stage
    lay = dec[dec.name.isin(LAYER_STAGES)]
    pl = lay.groupby(["layer", "name"]).ms.mean().unstack()
    for st in LAYER_STAGES:
        if st in pl:
            ax[2].plot(pl.index, pl[st], marker=".", label=st.replace("layer:", ""))
    ax[2].set_yscale("log")
    ax[2].set_title("Mean ms per layer call (log scale)")
    ax[2].set_xlabel("layer")
    ax[2].set_ylabel("ms")
    ax[2].legend(fontsize=7)

    # 4. attention time vs context length (mean over layers, per token position)
    mha = dec[dec.name == "layer:mha"].groupby("token_pos").ms.mean() * 1000  # us
    if len(mha) > 1:
        ax[3].plot(mha.index, mha.values, marker=".", lw=1, label="measured")
        slope, icpt = np.polyfit(mha.index, mha.values, 1)
        ax[3].plot(mha.index, slope * mha.index + icpt, "--", color="gray",
                   label=f"fit: {slope:.2f} us/token")
        ax[3].legend()
    ax[3].set_title("Attention (mha) time vs context length")
    ax[3].set_xlabel("token position (context length - 1)")
    ax[3].set_ylabel("us per layer call")
    ax[3].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
