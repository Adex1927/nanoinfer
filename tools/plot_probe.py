#!/usr/bin/env python3
"""
Plot the probe data written by nanoinfer (probe.csv).

CSV columns: name,phase,token_pos,layer,ms

Prints a text table and saves one figure:
  --phase decode (default) or --phase prefill: four plain panels (2x2):
    1. step latency vs token position          (trend / outliers)
    2. share of a step by stage                (what to optimize)
    3. mean time per layer, per stage          (are any layers odd?)
    4. per-token time of every stage vs context  (which ops grow with pos?)
  --phase both: six panels (3x2) comparing prefill and decode side-by-side.

Usage:
  python3 tools/plot_probe.py                                    # decode -> probe.png
  python3 tools/plot_probe.py --phase prefill -o prefill.png     # prefill -> prefill.png
  python3 tools/plot_probe.py --phase both -o probe_both.png     # prefill vs decode
  python3 tools/plot_probe.py run.csv -o run.png
  python3 tools/plot_probe.py --show
"""
import argparse

import numpy as np
import pandas as pd

# Stages that make up one token's work. Aggregates (all_layers,
# decode:forward_token, prefill:forward_token, prefill, prefill_total,
# decode:argmax) are excluded from the breakdown so nothing is double counted.
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


def per_token_breakdown(sub):
    """ms per token, per stage (layer stages summed over layers)."""
    parts = {}
    for st in LAYER_STAGES + TOKEN_STAGES:
        d = sub[sub.name == st]
        parts[st] = d.groupby("token_pos").ms.sum()
    return pd.DataFrame(parts).fillna(0.0)


def step_series(sub, bd, phase):
    """Return a Series of step latency indexed by token_pos."""
    probe_name = f"{phase}:forward_token"
    step = sub[sub.name == probe_name].sort_values("token_pos")
    if not step.empty:
        return step.set_index("token_pos").ms
    # Fallback for older CSVs without prefill:forward_token
    if not bd.empty:
        return bd.sum(axis=1).sort_index()
    return pd.Series(dtype=float)


def plot_single_phase(ax, sub, phase):
    label_cap = phase.capitalize()
    bd = per_token_breakdown(sub)
    s = step_series(sub, bd, phase)

    # 1. step latency vs token position
    if not s.empty:
        ax[0].plot(s.index, s.values, marker=".", lw=1)
        med = s.median()
        tok_s = 1000.0 / med if med > 0 else 0.0
        ax[0].axhline(med, color="gray", ls="--", lw=1,
                      label=f"median {med:.1f} ms ({tok_s:.1f} tok/s)")
        ax[0].legend()
    ax[0].set_title(f"{label_cap} step latency")
    ax[0].set_xlabel("token position")
    ax[0].set_ylabel("ms")
    ax[0].set_ylim(bottom=0)

    # 2. share of a step by stage
    if not bd.empty and bd.values.sum() > 0:
        mean_ms = bd.mean().sort_values()
        share = 100 * mean_ms / mean_ms.sum()
        ax[1].barh(mean_ms.index, mean_ms.values)
        for i, (ms, pct) in enumerate(zip(mean_ms.values, share.values)):
            ax[1].text(ms, i, f" {ms:.2f} ms ({pct:.1f}%)", va="center", fontsize=8)
        ax[1].set_xlim(right=mean_ms.max() * 1.35)
    ax[1].set_title(f"Mean ms per {phase} token, by stage")
    ax[1].set_xlabel("ms (layer stages summed over layers)")

    # 3. mean time per layer, per stage
    lay = sub[sub.name.isin(LAYER_STAGES)]
    if not lay.empty:
        pl = lay.groupby(["layer", "name"]).ms.mean().unstack()
        for st in LAYER_STAGES:
            if st in pl:
                ax[2].plot(pl.index, pl[st], marker=".", label=st.replace("layer:", ""))
        ax[2].set_yscale("log")
        ax[2].legend(fontsize=7)
    ax[2].set_title(f"{label_cap}: mean ms per layer call (log scale)")
    ax[2].set_xlabel("layer")
    ax[2].set_ylabel("ms")

    # 4. per-token time of every stage vs context length
    if len(bd) > 1:
        x = bd.index.values
        for st in bd.mean().sort_values(ascending=False).index:
            y = bd[st].values
            if not (y > 0).any():
                continue
            slope_us = np.polyfit(x, y, 1)[0] * 1000  # ms/pos -> us/pos
            label = st.replace("layer:", "")
            ax[3].plot(x, y, lw=1, label=f"{label} ({slope_us:+.2f} us/pos)")
        ax[3].set_yscale("log")
        ax[3].legend(fontsize=7)
    ax[3].set_title(f"{label_cap}: per-token stage time vs context (log scale)")
    ax[3].set_xlabel("token position (context length - 1)")
    ax[3].set_ylabel("ms per token (summed over layers)")


def plot_both_phases(ax, df):
    pre = df[df.phase == "prefill"]
    dec = df[df.phase == "decode"]
    bd_pre = per_token_breakdown(pre)
    bd_dec = per_token_breakdown(dec)
    s_pre = step_series(pre, bd_pre, "prefill")
    s_dec = step_series(dec, bd_dec, "decode")

    # 1. Step latency across whole trajectory (prefill + decode)
    if not s_pre.empty:
        med = s_pre.median()
        ax[0].plot(s_pre.index, s_pre.values, marker=".", lw=1,
                   label=f"prefill (med {med:.1f} ms, {1000.0/med:.1f} tok/s)")
    if not s_dec.empty:
        med = s_dec.median()
        ax[0].plot(s_dec.index, s_dec.values, marker=".", lw=1,
                   label=f"decode (med {med:.1f} ms, {1000.0/med:.1f} tok/s)")
    if not s_pre.empty or not s_dec.empty:
        ax[0].legend(fontsize=8)
    ax[0].set_title("Prefill vs Decode step latency")
    ax[0].set_xlabel("token position")
    ax[0].set_ylabel("ms per token")
    ax[0].set_ylim(bottom=0)

    # 2. Side-by-side grouped horizontal bar chart: mean ms/token by stage
    stages = ( bd_dec.mean().sort_values().index if not bd_dec.empty
               else bd_pre.mean().sort_values().index )
    y_pos = np.arange(len(stages))
    height = 0.38
    pre_vals = bd_pre.mean().reindex(stages, fill_value=0.0).values if not bd_pre.empty else np.zeros(len(stages))
    dec_vals = bd_dec.mean().reindex(stages, fill_value=0.0).values if not bd_dec.empty else np.zeros(len(stages))
    ax[1].barh(y_pos + height / 2, pre_vals, height, label="prefill")
    ax[1].barh(y_pos - height / 2, dec_vals, height, label="decode")
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(stages)
    max_v = max(pre_vals.max(initial=0.0), dec_vals.max(initial=0.0))
    if max_v > 0:
        ax[1].set_xlim(right=max_v * 1.25)
    ax[1].legend(fontsize=8)
    ax[1].set_title("Mean ms per token by stage (Prefill vs Decode)")
    ax[1].set_xlabel("ms (layer stages summed over layers)")

    # 3..6: Prefill and Decode layer breakdown + context scaling
    for idx_offset, (sub, phase) in enumerate([(pre, "prefill"), (dec, "decode")]):
        a_lay = ax[2 + idx_offset * 2]
        a_ctx = ax[3 + idx_offset * 2]
        bd = bd_pre if phase == "prefill" else bd_dec
        label_cap = phase.capitalize()

        lay = sub[sub.name.isin(LAYER_STAGES)]
        if not lay.empty:
            pl = lay.groupby(["layer", "name"]).ms.mean().unstack()
            for st in LAYER_STAGES:
                if st in pl:
                    a_lay.plot(pl.index, pl[st], marker=".", label=st.replace("layer:", ""))
            a_lay.set_yscale("log")
            a_lay.legend(fontsize=7)
        a_lay.set_title(f"{label_cap}: mean ms per layer call (log scale)")
        a_lay.set_xlabel("layer")
        a_lay.set_ylabel("ms")

        if len(bd) > 1:
            x = bd.index.values
            for st in bd.mean().sort_values(ascending=False).index:
                y = bd[st].values
                if not (y > 0).any():
                    continue
                slope_us = np.polyfit(x, y, 1)[0] * 1000
                label = st.replace("layer:", "")
                a_ctx.plot(x, y, lw=1, label=f"{label} ({slope_us:+.2f} us/pos)")
            a_ctx.set_yscale("log")
            a_ctx.legend(fontsize=7)
        a_ctx.set_title(f"{label_cap}: per-token stage time vs context (log scale)")
        a_ctx.set_xlabel("token position")
        a_ctx.set_ylabel("ms per token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="probe.csv")
    ap.add_argument("-o", "--out", default="probe.png")
    ap.add_argument("--phase", choices=["decode", "prefill", "both"], default="decode",
                    help="Which generation phase to plot (default: decode)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    print_table(df)

    if args.phase == "both":
        fig, axs = plt.subplots(3, 2, figsize=(13, 13))
        plot_both_phases(axs.ravel(), df)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        sub = df[df.phase == args.phase]
        plot_single_phase(axs.ravel(), sub, args.phase)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
