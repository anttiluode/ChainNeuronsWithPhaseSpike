"""What does a 2 pi slip emit? (Sol's question after relay3.) Physics unchanged.

One near-complement pair (delta=0.015) at (24,24), pulse source at (24,26.5) on a 0.11 carrier.
Sub-threshold (0.04) vs supra-threshold (0.06) pulses differ by 0.02; a second sub-threshold step (0.02 -> 0.04)
of the same size is the linear baseline. The pair's own field = field(with pair) - field(bare medium, same pulse).
Also records how the 2 pi of S is shared between A and X (which decides whether the emission keeps a sign flip).
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_num_threads(1)
from chains.fork import fork_world
from chains.probing import source_kernel
from experiments.theta_neuron import world, PULSE_FREQ, PULSE_DUR

PROBES = [(24, 24 - r) for r in (3, 6, 9, 12, 16)]   # opposite side from the source


def run(w0, amp, steps=2600, start=100, bare=False):
    w = fork_world(w0)
    if bare:
        w.cfg.g_emit = 0.0
    k = source_kernel(w, (24.0, 26.5), 1.0)
    prev = w.theta.numpy().copy(); un = np.zeros(2)
    th, fld, emit = [], [], []
    for t in range(steps):
        a = amp * math.cos(PULSE_FREQ * int(w.t)) if start <= t < start + PULSE_DUR else 0.0
        w.step(external_source=(k * a) if a else None)
        cur = w.theta.numpy(); un += (cur - prev + math.pi) % (2 * math.pi) - math.pi; prev = cur.copy()
        th.append(un.copy()); emit.append(np.cos(cur).sum())
        p = w.psi.numpy(); fld.append([p[y, x] for x, y in PROBES])
    return np.array(th), np.array(fld), np.array(emit)


def main(out="results/theta/slip_radiation.json"):
    w0 = world(0.015, [(24.0, 24.0)])
    R = {}
    for amp in (0.02, 0.04, 0.06, 0.2):
        th, f, e = run(w0, amp)
        _, fb, _ = run(w0, amp, bare=True)
        R[amp] = dict(th=th, pair=f - fb, emit=e)
    t = np.arange(2600)
    res = {"probes_dist": [3, 6, 9, 12, 16]}
    # how S's winding is split
    base = R[0.02]["th"]
    for amp in (0.04, 0.06, 0.2):
        d = R[amp]["th"][-1] - base[-1]
        res[f"winding_split_amp{amp}"] = {"dthetaA": float(d[0]), "dthetaX": float(d[1]),
                                          "dS": float(d.sum()), "dD": float(d[0] - d[1])}
    # event field vs linear step, early (during/after slip: 100-900) and late (1600-2600)
    def rms(x, a, b): return np.sqrt((x[a:b] ** 2).mean(0))
    ev = R[0.06]["pair"] - R[0.04]["pair"]; lin = R[0.04]["pair"] - R[0.02]["pair"]
    res["event_rms_early"] = rms(ev, 100, 900).tolist(); res["linear_rms_early"] = rms(lin, 100, 900).tolist()
    res["event_rms_late"] = rms(ev, 1600, 2600).tolist(); res["linear_rms_late"] = rms(lin, 1600, 2600).tolist()
    res["pair_field_rms_late"] = rms(R[0.04]["pair"], 1600, 2600).tolist()
    # does the late emission flip sign? correlation of late emission supra vs sub
    e1, e0 = R[0.06]["emit"][1600:], R[0.04]["emit"][1600:]
    res["late_emission_corr_supra_vs_sub"] = float(np.corrcoef(e1, e0)[0, 1])
    # spectrum of the event field at the nearest probe: fraction of power above the gap (0.1414)
    x = ev[100:1600, 0]; F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = 2 * math.pi * np.fft.rfftfreq(len(x))
    res["event_power_frac_above_gap"] = float(F[fr > 0.1414].sum() / F.sum())
    res["event_peak_freq"] = float(fr[F.argmax()])
    json.dump(res, open(out, "w"), indent=1)
    np.savez("results/theta/slip_radiation_traces.npz", **{f"{k}_{a}": v[k] for a, v in R.items() for k in ("th", "pair", "emit")})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
