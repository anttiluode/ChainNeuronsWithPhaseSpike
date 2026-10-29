"""Figures for the README: phase spikes moved by memory, and transmission vs distance."""
import json
import math
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


def theta_figure(out="results/fig_theta_neuron.png"):
    th = json.load(open("results/theta/threshold.json"))["threshold"]
    win = json.load(open("results/theta/window.json"))["window"]
    gate = json.load(open("results/theta/gate.json"))["gate"]
    edge = json.load(open("results/theta/edge.json"))["edge"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for key, lab in (("0.015", "near-complement δ=0.015"), ("0.0", "exact complement δ=0")):
        r = th[key]
        ax[0].plot([x["amp"] for x in r], [x["plus"]["peak"] for x in r], "o-", label=lab)
    ax[0].axhline(2 * math.pi, color="k", lw=0.5, ls=":"); ax[0].set_xlabel("pulse amplitude")
    ax[0].set_ylabel("peak sum-phase excursion (rad)"); ax[0].legend(fontsize=7)
    ax[0].set_title("threshold and all-or-none 2π slip", fontsize=9)
    ax[1].bar([x["carrier_phase"] for x in win], [x["slips"] for x in win], width=0.35)
    ax[1].set_xlabel("trigger carrier phase (rad)"); ax[1].set_ylabel("spikes"); ax[1].set_title("firing window", fontsize=9)
    ax[2].scatter([g["retained_dD"] for g in gate], [g["trigger_slips"] for g in gate], c=["C2" if g["trigger_slips"] else "C3" for g in gate])
    ax[2].set_xlabel("stored difference phase D (rad)"); ax[2].set_ylabel("same trigger fires?")
    ax[2].set_title("memory decides what fires it", fontsize=9)
    ax[3].plot([e["delta"] for e in edge], [max(e["slip_rate_rad_per_step"], 0) for e in edge], "o-")
    ax[3].set_xlabel("detuning δ"); ax[3].set_ylabel("spontaneous slip rate"); ax[3].set_title("locking edge (tonic beyond)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


if __name__ == "__main__":
    spike_figure()
    transmission_figure()
    theta_figure()
