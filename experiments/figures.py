"""Figures for the README: phase spikes moved by memory, and transmission vs distance."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chains.fork import fork_world
from chains.neuron import T_SPIKE, build, record, spikes, write


def spike_figure(out="results/fig_phase_spike.png"):
    base, groups = build(N=64, chains=(("ACGT", (32.0, 32.0)),))
    fig, ax = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for k, bits in enumerate(([1, 1], [1, -1], [-1, 1], [-1, -1])):
        w = fork_world(base)
        write(w, (32.0, 34.5), bits)
        for _ in range(2000):
            w.step()
        sig, ts = record(w, groups, 1000)
        st, h, env = spikes(sig[0], ts)
        t = ts - ts[0]
        ax[k].plot(t, sig[0], lw=0.6, color="0.6", label="chain output  sum cos(theta)")
        ax[k].plot(t, env, color="C3", lw=1.2, label="envelope")
        for s in st:
            ax[k].axvline(s - ts[0], color="C0", lw=1.0)
        ax[k].set_ylabel(f"bits {bits}", fontsize=8)
        ax[k].set_ylim(-4.5, 4.5)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("One ACGT chain, 2000 steps after a write: the bursts (blue) recur every 2π/0.02 ≈ 314 steps; "
                    "their timing is set by what was written", fontsize=9)
    ax[-1].set_xlabel("steps (common pump clock)")
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


def transmission_figure(out="results/fig_transmission.png"):
    ds, sr, ro, sx, rx = [], [], [], [], []
    for d in (3, 5, 8, 20):
        p = f"results/s2_d{d}/summary.json"
        if not os.path.exists(p):
            continue
        c = json.load(open(p))["conditions"]
        ds.append(d)
        sr.append(c["sender+receiver"]["pattern_4way"]); sx.append(c["sender+receiver"]["b1*b2"])
        ro.append(c.get("receiver_only", {}).get("pattern_4way", np.nan))
        rx.append(c.get("receiver_only", {}).get("b1*b2", np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(ds, sr, "o-", label="sender present"); ax[0].plot(ds, ro, "s--", label="receiver alone (leakage)")
    ax[0].axhline(0.25, color="k", lw=0.5); ax[0].set_ylabel("4-way pattern from receiver spike timing")
    ax[1].plot(ds, sx, "o-", label="sender present"); ax[1].plot(ds, rx, "s--", label="receiver alone (leakage)")
    ax[1].axhline(0.5, color="k", lw=0.5); ax[1].set_ylabel("sender's b1·b2 from receiver spike timing")
    for a in ax:
        a.set_xlabel("chain separation (cells)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


if __name__ == "__main__":
    spike_figure()
    transmission_figure()
