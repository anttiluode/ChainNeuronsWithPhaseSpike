"""S3: does experience build a better place to store the same experience? (Astra's cross-over)

GROW  two soups from the same inventory and type order (4 x 80 monomers, seed-matched), default physics,
      under a spatially uniform carrier at A = 0.03 or B = 0.09 (amplitude 0.002) for --grow steps.
SET   pin every monomer, stop ligation and noise. ERASE the growth signal: field psi = psi_prev = 0 and the
      SAME uniform random phase draw in both arms (so no residual written phase survives), then relock.
TEST  [labelled change] phase coupling g_phase is lowered to --test-gphase for the test only: at the default
      0.02 a frozen soup is chaotic (ComputationalChains driven_stability: +0.033/step), so + and - writes
      decorrelate regardless of memory (smoke test: R = pi/2, the value for unrelated phases). A chaos floor
      (two copies differing by 1e-6 rad on one monomer, no write) is reported alongside.
      Identical challenges: uniform BPSK write (+/-) at A and at B, amplitude --amp for 600 steps, then 600
      silent steps. Response R = mean over monomers of |wrap(theta(+) - theta(-))| = how strongly that carrier
      writes the soup in a way that persists.
SI    selectivity = [R(A | grown A) - R(A | grown B)] - [R(B | grown A) - R(B | grown B)]  (> 0 means each soup
      responds more to its own carrier).
VARIANTS of the frozen soup, to locate the record: as grown | bonds cut (positions kept) | chain_boost 0
      (emission matched) | positions scrambled (arrangement removed, bonds cut).

    python experiments/crossover.py --seeds 3 --grow 6000
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

CARRIERS = {"A": 0.03, "B": 0.09}


def grow(seed, freq, args):
    w = World(Config(N=args.N, seed=seed, device="cpu"))
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(4), args.n_each)
    rng.shuffle(types)
    w.add_monomers(types)
    ones = torch.ones(args.N, args.N)
    for t in range(args.grow):
        w.step(external_source=ones * float(args.grow_amp * math.cos(freq * t)))
    return w


def composition(w):
    typ = w.typ.numpy()
    bonded = (w.nb >= 0).any(1).numpy()
    return {w.letters[k]: float(bonded[typ == k].mean()) for k in range(4)}, int(w.ligations)


def prepare(w, seed, variant, args):
    f = fork_world(w)
    f.pinned[:] = True
    f.cfg.p_lig, f.cfg.kT, f.cfg.phase_noise = 0.0, 0.0, 0.0
    if variant in ("bonds_cut", "scrambled"):
        f.nb.fill_(-1)
    if variant == "emission_matched":
        f.cfg.chain_boost = 0.0
    if variant == "scrambled":
        g = torch.Generator().manual_seed(seed + 99)
        f.pos = torch.rand(f.pos.shape, generator=g) * args.N
    f.cfg.g_phase = args.test_gphase
    f.psi.zero_(); f.psi_prev.zero_()
    g = torch.Generator().manual_seed(seed + 7)
    f.theta = torch.rand(f.P, generator=g, dtype=f.theta.dtype) * 2 * math.pi
    f.held_by.zero_()
    for _ in range(args.relock):
        f.step()
    return f


def response(base, freq, args):
    ones = torch.ones(args.N, args.N)
    out = {}
    for sgn in (1, -1):
        v = fork_world(base)
        for _ in range(600):
            v.step(external_source=ones * float(sgn * args.amp * math.cos(freq * int(v.t))))
        for _ in range(600):
            v.step()
        out[sgn] = v.theta.clone()
    d = torch.remainder(out[1] - out[-1] + math.pi, 2 * math.pi) - math.pi
    return float(d.abs().mean())


def chaos_floor(base):
    a, b = fork_world(base), fork_world(base)
    b.theta[0] += 1e-6
    for _ in range(1200):
        a.step(); b.step()
    d = torch.remainder(a.theta - b.theta + math.pi, 2 * math.pi) - math.pi
    return float(d.abs().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--n-each", type=int, default=80)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--grow", type=int, default=6000)
    p.add_argument("--grow-amp", type=float, default=0.002)
    p.add_argument("--relock", type=int, default=800)
    p.add_argument("--amp", type=float, default=0.002)
    p.add_argument("--test-gphase", type=float, default=0.0005)
    p.add_argument("--variants", nargs="+", default=["as_grown", "bonds_cut", "emission_matched", "scrambled"])
    p.add_argument("--out", default="results/s3_crossover")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        grown = {arm: grow(seed, f, a) for arm, f in CARRIERS.items()}
        comp = {arm: composition(w) for arm, w in grown.items()}
        print(f"seed {seed} composition {comp}", flush=True)
        for var in a.variants:
            R = {}
            floors = {}
            for arm, w in grown.items():
                base = prepare(w, seed, var, a)
                floors[arm] = chaos_floor(base)
                for test, f in CARRIERS.items():
                    R[(arm, test)] = response(base, f, a)
            si = (R[("A", "A")] - R[("B", "A")]) - (R[("A", "B")] - R[("B", "B")])
            row = {"seed": seed, "variant": var, "SI": si, "composition": comp, "chaos_floor": floors,
                   **{f"R_test{t}_grown{g}": v for (g, t), v in R.items()}}
            rows.append(row)
            print(f"  {var:16s} R(testA|grownA)={R[('A','A')]:.4f} R(testA|grownB)={R[('B','A')]:.4f} "
                  f"R(testB|grownA)={R[('A','B')]:.4f} R(testB|grownB)={R[('B','B')]:.4f}  SI={si:+.4f}  "
                  f"floor A {floors['A']:.1e} B {floors['B']:.1e}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for var in a.variants:
        s = [r["SI"] for r in rows if r["variant"] == var]
        print(f"{var:16s} SI per seed {np.round(s, 4).tolist()}  mean {np.mean(s):+.4f}  positive in {sum(x > 0 for x in s)}/{len(s)}")


if __name__ == "__main__":
    main()
