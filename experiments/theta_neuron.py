"""T: a near-complement pair is a theta neuron (class-1 excitable) in the UNMODIFIED physics.

The phase equation in chains/world.py is Adler-type. For a pair A (0.03) + X (0.11 + delta), the pump-averaged sum phase
S = theta_A + theta_X - 2 Omega t obeys dS/dt ~ delta - K sin(S - s0). Exact complements (delta = 0) sit deep in
locking. For delta just below the locking edge delta_c, the locked state is the ghost of a saddle-node on a circle:
small kicks relax, kicks past the saddle produce one full 2 pi slip and return -- a spike with a threshold, an
all-or-none size and a refractory time. Above delta_c the pair slips periodically (tonic firing) at a rate that
should scale like sqrt(delta - delta_c).

Only frequency choice and stimulus differ from the parent physics (letter X added to the alphabet); no rule added.

Tests
  edge       slip rate vs delta (locking edge, sqrt law)
  threshold  net S change vs pulse amplitude at delta = 0.015 (and 0.0 as deep-lock control)
  window     firing vs pulse carrier phase (which input phases trigger)
  refractory two supra-threshold pulses separated by dt: does the second fire?
  relay      sender pair fired by a pulse; does a receiver pair d cells away fire? control: sender absent

    python experiments/theta_neuron.py --tests edge threshold window refractory relay
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

from chains import Config, World
from chains.fork import fork_world
from chains.probing import source_kernel

PAIR_DIST = 2.4
N = 48
PULSE_FREQ, PULSE_DUR = 0.11, 80     # carrier at A's complement: pushes theta_A


def world(delta, pairs, noise=0.0, warm=3000, deltas=None):
    """pairs: list of (x_center, y). Each pair is A at x-1.2 and X_k at x+1.2. deltas: per-pair detuning (optional)."""
    deltas = deltas or [delta] * len(pairs)
    alph = {"A": 0.03, "C": 0.05, "G": 0.09, "T": 0.11}
    for k, dk in enumerate(sorted(set(deltas))):
        alph[f"X{k}"] = 0.11 + dk
    cfg = Config(N=N, seed=0, device="cpu", p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0, alphabet=alph)
    w = World(cfg)
    keys = sorted(set(deltas))
    for (x, y), dk in zip(pairs, deltas):
        w.add_monomers([0, w.letters.index(f"X{keys.index(dk)}")], pos=[(x - PAIR_DIST / 2, y), (x + PAIR_DIST / 2, y)])
    w.pinned[:] = True
    for _ in range(warm):
        w.step()
    w.cfg.phase_noise = noise
    return w


class SumPhase:
    """Unwrapped sum phase relative to the pump for each pair (beads 2k, 2k+1)."""

    def __init__(self, w):
        self.prev = w.theta.numpy().copy()
        self.un = np.zeros_like(self.prev)
        self.t0 = int(w.t)

    def update(self, w):
        th = w.theta.numpy()
        self.un += (th - self.prev + math.pi) % (2 * math.pi) - math.pi
        self.prev = th.copy()
        p = self.un.reshape(-1, 2).sum(1)
        return p - 0.14 * (int(w.t) - self.t0)


def drive(w, src, pulses, steps, phase=0.0, seed=None):
    """pulses: (start, amplitude). Returns S trajectories (steps, n_pairs), relative to start."""
    w = fork_world(w)
    if seed is not None:
        w.gen.manual_seed(seed)
    k = source_kernel(w, src, 1.0)
    sp = SumPhase(w)
    out = []
    for t in range(steps):
        a = sum(A * math.cos(PULSE_FREQ * int(w.t) + phase) for (s, A) in pulses if s <= t < s + PULSE_DUR)
        w.step(external_source=(k * float(a)) if a else None)
        out.append(sp.update(w))
    out = np.array(out)
    return out - out[0]


def slips(S_final):
    return int(np.round(S_final / (2 * math.pi)))


def t_edge(res):
    rows = []
    for d in (0.010, 0.014, 0.015, 0.0155, 0.016, 0.017, 0.018, 0.019, 0.020, 0.022, 0.025):
        w = world(d, [(24.0, 24.0)])
        S = drive(w, (24.0, 26.5), [], 6000)[:, 0]
        late = S[3000:]
        rate = float(np.polyfit(np.arange(len(late)), late, 1)[0])
        rows.append({"delta": d, "slip_rate_rad_per_step": rate})
        print(f"edge delta={d:.4f} rate={rate:+.5f}", flush=True)
    res["edge"] = rows


def t_threshold(res):
    out = {}
    for d in (0.015, 0.0):
        w = world(d, [(24.0, 24.0)])
        rows = []
        for A in (0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            r = {}
            for sgn in (1, -1):
                S = drive(w, (24.0, 26.5), [(100, sgn * A)], 2500)[:, 0]
                r[sgn] = {"net": float(S[-1]), "peak": float(np.abs(S).max()), "slips": slips(S[-1])}
            rows.append({"amp": A, "plus": r[1], "minus": r[-1]})
            print(f"threshold delta={d} amp={A:.2f} +: slips {r[1]['slips']} peak {r[1]['peak']:.2f} | "
                  f"-: slips {r[-1]['slips']} peak {r[-1]['peak']:.2f}", flush=True)
        out[str(d)] = rows
    res["threshold"] = out


def t_window(res):
    w = world(0.015, [(24.0, 24.0)])
    rows = []
    for ph in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        S = drive(w, (24.0, 26.5), [(100, 0.2)], 2500, phase=float(ph))[:, 0]
        rows.append({"carrier_phase": float(ph), "slips": slips(S[-1]), "peak": float(np.abs(S).max())})
    res["window"] = rows
    print("window (carrier phase -> slips):", [(round(r["carrier_phase"], 2), r["slips"]) for r in rows], flush=True)


def t_refractory(res):
    w = world(0.015, [(24.0, 24.0)])
    rows = []
    for dt in (100, 150, 200, 300, 400, 600, 900, 1300, 1800):
        S = drive(w, (24.0, 26.5), [(100, 0.2), (100 + dt, 0.2)], 2600 + dt)[:, 0]
        rows.append({"dt": dt, "slips": slips(S[-1])})
        print(f"refractory dt={dt}: total slips {slips(S[-1])}", flush=True)
    res["refractory"] = rows


def t_relay(res):
    rows = []
    for d in (3.0, 4.0, 5.0, 7.0):
        for arm in ("sender+receiver", "receiver_only"):
            pairs = [(24.0, 24.0 - d / 2), (24.0, 24.0 + d / 2)] if arm == "sender+receiver" else [(24.0, 24.0 + d / 2)]
            w = world(0.015, pairs)
            src = (24.0, 24.0 - d / 2 - 2.5)     # far side of the sender
            for A in (0.0, 0.2):
                S = drive(w, src, [(100, A)], 3000)
                rec = S[:, -1]
                row = {"dist": d, "arm": arm, "amp": A, "receiver_slips": slips(rec[-1]),
                       "receiver_peak": float(np.abs(rec).max()),
                       "sender_slips": slips(S[-1, 0]) if arm == "sender+receiver" else None}
                if row["receiver_slips"] or A > 0:
                    print(f"relay d={d} {arm:16s} amp={A}: sender slips {row['sender_slips']} receiver slips "
                          f"{row['receiver_slips']} (peak {row['receiver_peak']:.2f})", flush=True)
                rows.append(row)
    res["relay"] = rows


def t_gate(res):
    """Memory-gated firing. Write the pair's free DIFFERENCE phase D with a carrier at X's complement
    (2 Omega - w_X = 0.015 for delta 0.015) of varying signed amplitude; S relaxes back (locked), D keeps the shift.
    Then the SAME trigger (0.11 carrier, amp 0.15, fixed phase) 1500 steps later. Fire or not, vs retained D."""
    d = 0.015
    w = world(d, [(24.0, 24.0)])
    wfreq = 0.14 - (0.11 + d)
    rows = []
    for wa in np.linspace(-0.4, 0.4, 17):
        v = fork_world(w)
        k = source_kernel(v, (24.0, 26.5), 1.0)
        th0 = v.theta.numpy().copy()
        sp = SumPhase(v)
        for t in range(300):
            v.step(external_source=k * float(wa * math.cos(wfreq * int(v.t)))); sp.update(v)
        for t in range(1500):
            v.step(); S_pre = sp.update(v)
        th1 = v.theta.numpy()
        dD = float(np.angle(np.exp(1j * ((th1[0] - th1[1]) - (th0[0] - th0[1])))))
        written_slips = slips(S_pre[0])
        S = drive(v, (24.0, 26.5), [(100, 0.15)], 2500)[:, 0]
        rows.append({"write_amp": float(wa), "retained_dD": dD, "slips_during_write": written_slips,
                     "trigger_slips": slips(S[-1]), "trigger_peak": float(np.abs(S).max())})
        print(f"gate write {wa:+.2f}: retained dD {dD:+.2f}, slips while writing {written_slips}, "
              f"trigger -> slips {slips(S[-1])} (peak {np.abs(S).max():.2f})", flush=True)
    res["gate"] = rows


def t_relay2(res):
    """Relay redesigned: weaker detuning so coupled pairs stay quiet, weak pulse so leakage stays sub-threshold."""
    rows = []
    for d_tune in (0.008, 0.011, 0.013):
        for dist in (3.0, 4.0, 5.0):
            for A in (0.08, 0.12, 0.2):
                rec = {}
                for arm in ("sender+receiver", "receiver_only", "sender+receiver_nopulse"):
                    two = arm.startswith("sender")
                    pairs = [(24.0, 24.0 - dist / 2), (24.0, 24.0 + dist / 2)] if two else [(24.0, 24.0 + dist / 2)]
                    w = world(d_tune, pairs)
                    amp = 0.0 if arm.endswith("nopulse") else A
                    S = drive(w, (24.0, 24.0 - dist / 2 - 2.5), [(100, amp)], 3000)
                    rec[arm] = {"receiver": slips(S[-1, -1]), "sender": slips(S[-1, 0]) if two else None}
                row = {"delta": d_tune, "dist": dist, "amp": A, **rec}
                valid = (rec["sender+receiver_nopulse"]["receiver"] == 0 and rec["sender+receiver_nopulse"]["sender"] == 0
                         and rec["receiver_only"]["receiver"] == 0 and rec["sender+receiver"]["sender"] != 0)
                row["valid_design"] = bool(valid)
                row["relayed"] = bool(valid and rec["sender+receiver"]["receiver"] != 0)
                rows.append(row)
                print(f"relay2 delta={d_tune} d={dist} amp={A}: quiet={rec['sender+receiver_nopulse']} "
                      f"leak={rec['receiver_only']['receiver']} with-sender={rec['sender+receiver']} "
                      f"valid={valid} relayed={row['relayed']}", flush=True)
    res["relay2"] = rows


def t_relay3(res):
    """Refined relay: equal detuning in the 0.011-0.013 gap and UNEQUAL detuning (receiver quieter)."""
    rows = []
    combos = [(0.0115, 0.0115), (0.012, 0.012), (0.0125, 0.0125), (0.015, 0.008), (0.015, 0.011), (0.014, 0.010)]
    for ds, dr in combos:
        for dist in (3.0, 4.0):
            for A in (0.15, 0.3, 0.45):
                rec = {}
                for arm in ("sender+receiver", "receiver_only", "sender+receiver_nopulse"):
                    two = arm.startswith("sender")
                    pairs = [(24.0, 24.0 - dist / 2), (24.0, 24.0 + dist / 2)] if two else [(24.0, 24.0 + dist / 2)]
                    w = world(None, pairs, deltas=[ds, dr] if two else [dr])
                    amp = 0.0 if arm.endswith("nopulse") else A
                    S = drive(w, (24.0, 24.0 - dist / 2 - 2.5), [(100, amp)], 3000)
                    rec[arm] = {"receiver": slips(S[-1, -1]), "sender": slips(S[-1, 0]) if two else None}
                valid = (rec["sender+receiver_nopulse"]["receiver"] == 0 and rec["sender+receiver_nopulse"]["sender"] == 0
                         and rec["receiver_only"]["receiver"] == 0 and rec["sender+receiver"]["sender"] != 0)
                row = {"delta_sender": ds, "delta_receiver": dr, "dist": dist, "amp": A, **rec,
                       "valid_design": bool(valid), "relayed": bool(valid and rec["sender+receiver"]["receiver"] != 0)}
                rows.append(row)
                print(f"relay3 ds={ds} dr={dr} d={dist} amp={A}: quiet={rec['sender+receiver_nopulse']} "
                      f"leak={rec['receiver_only']['receiver']} with-sender={rec['sender+receiver']} "
                      f"valid={valid} relayed={row['relayed']}", flush=True)
    res["relay3"] = rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", nargs="+", default=["edge", "threshold", "window", "refractory", "relay"])
    p.add_argument("--out", default="results/theta")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for t in a.tests:
        res = {}
        globals()[f"t_{t}"](res)
        json.dump(res, open(os.path.join(a.out, f"{t}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
