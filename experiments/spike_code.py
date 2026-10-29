"""S1 and S2: what a chain-neuron's spike timing carries.

S1 (one chain): write 2 bits (BPSK, 0.03 -> A/T pair, 0.09 -> C/G pair), wait, then read ONLY the timing of
the chain's spikes relative to the pump clock (one circular number per trial). Decode the bits from it.
S2 (two chains): the same write is aimed at a SENDER chain; a RECEIVER chain sits d cells away. Read the
receiver's spike timing and decode the sender's bits from it. Control: the same source with the sender removed
(receiver-only), which measures direct leakage of the evanescent write onto the receiver. Also report spike
synchrony between sender and receiver.

Trials share a warmed start (a reset) and differ by phase noise; 8-fold CV, ridge on (cos, sin) of spike phase.

    python experiments/spike_code.py --mode s1 --trials 64
    python experiments/spike_code.py --mode s2 --dist 3 --trials 48
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains.fork import fork_world
from chains.neuron import T_SPIKE, build, record, spike_phase, spikes, write


def ridge_cv(X, y, alpha=1e-3, folds=8):
    n = len(y)
    idx = np.arange(n)
    hits = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        B = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd])
        coef = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])
        hits += int(np.sum(np.sign(B @ coef) == y[te]))
    return hits / n


def centroid_cv(X, labels, folds=8):
    """Nearest-centroid classification of the 4 patterns (multi-class)."""
    n = len(labels)
    idx = np.arange(n)
    hits = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        cents = {c: X[tr][labels[tr] == c].mean(0) for c in np.unique(labels[tr])}
        for i in te:
            pred = min(cents, key=lambda c: np.sum((X[i] - cents[c]) ** 2))
            hits += int(pred == labels[i])
    return hits / n


def features(phases):
    return np.column_stack([np.cos(phases), np.sin(phases)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("s1", "s2"), default="s1")
    p.add_argument("--dist", type=float, default=3.0)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--phase-noise", type=float, default=0.003)
    p.add_argument("--delay", type=int, default=2000)
    p.add_argument("--read", type=int, default=1300)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rng = np.random.default_rng(5)
    pats = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]] * (a.trials // 4 + 1))[: a.trials].astype(float)
    pats = pats[rng.permutation(len(pats))]
    labels = (pats[:, 0] > 0) * 2 + (pats[:, 1] > 0)
    N, y0 = 64, 32.0
    if a.mode == "s1":
        layouts = {"chain": [("ACGT", (32.0, y0))]}
        src = (32.0, y0 + 2.5)
    else:
        ys, yr = y0 - a.dist / 2, y0 + a.dist / 2
        layouts = {"sender+receiver": [("ACGT", (32.0, ys)), ("ACGT", (32.0, yr))],
                   "receiver_only": [("ACGT", (32.0, yr))]}
        src = (32.0, ys - 2.5)                      # on the far side of the sender
    res = {"mode": a.mode, "dist": a.dist, "trials": a.trials, "phase_noise": a.phase_noise,
           "delay": a.delay, "read": a.read, "T_spike": T_SPIKE, "conditions": {}}
    for name, chains in layouts.items():
        base, groups = build(N=N, chains=chains, phase_noise=a.phase_noise)
        recv = len(groups) - 1                    # the last chain is the one we read (receiver in S2)
        phases, heights, sync = [], [], []
        for i, bits in enumerate(pats):
            w = fork_world(base)
            w.gen.manual_seed(1000 + i)
            write(w, src, bits)
            for _ in range(a.delay):
                w.step()
            sig, ts = record(w, groups, a.read)
            per_chain = []
            for j in range(len(groups)):
                st, sh, _ = spikes(sig[j], ts)
                per_chain.append((spike_phase(st), sh.mean() if len(sh) else 0.0))
            (ph, R), h = per_chain[recv]
            phases.append(ph); heights.append(h)
            if len(groups) == 2:
                d = per_chain[0][0][0] - per_chain[1][0][0]
                sync.append(float(np.angle(np.exp(1j * d))))
        phases = np.array(phases)
        X = features(phases)
        cond = {"b1": ridge_cv(X, pats[:, 0]), "b2": ridge_cv(X, pats[:, 1]),
                "b1*b2": ridge_cv(X, pats[:, 0] * pats[:, 1]), "pattern_4way": centroid_cv(X, labels),
                "spike_phase": phases.tolist(), "spike_height": heights,
                "per_pattern_mean_phase": {str(int(c)): float(np.angle(np.exp(1j * phases[labels == c]).mean()))
                                           for c in range(4)},
                "per_pattern_R": {str(int(c)): float(abs(np.exp(1j * phases[labels == c]).mean())) for c in range(4)}}
        if sync:
            s = np.array(sync)
            cond["sender_minus_receiver_phase"] = s.tolist()
            cond["sync_R"] = float(abs(np.exp(1j * s).mean()))
        res["conditions"][name] = cond
        print(f"{a.mode} d={a.dist:g} {name:16s} decode from spike timing: b1={cond['b1']:.2f} b2={cond['b2']:.2f} "
              f"b1*b2={cond['b1*b2']:.2f} 4-way={cond['pattern_4way']:.2f} (chance .25)  "
              f"per-pattern phase {[round(v, 2) for v in cond['per_pattern_mean_phase'].values()]} "
              f"R {[round(v, 2) for v in cond['per_pattern_R'].values()]}"
              + (f"  sync R={cond['sync_R']:.2f}" if sync else ""), flush=True)
    out = a.out or f"results/{a.mode}" + (f"_d{a.dist:g}" if a.mode == "s2" else "")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
