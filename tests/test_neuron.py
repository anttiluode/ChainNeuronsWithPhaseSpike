"""Spike instruments, plus the physical claim that a chain's output bursts with period 2 pi / 0.02."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from chains.neuron import T_SPIKE, build, record, spike_phase, spikes


def test_spike_detection_on_synthetic_beats():
    t = np.arange(3000)
    sig = sum(np.cos(w * t + 0.3 * k) for k, w in enumerate((0.03, 0.05, 0.09, 0.11)))
    st, h, env = spikes(sig, t)
    gaps = np.diff(st)
    assert len(st) >= 8 and np.allclose(gaps, T_SPIKE, atol=3)


def test_spike_phase_is_circular():
    ph, R = spike_phase([0.0, T_SPIKE, 2 * T_SPIKE])
    assert abs(ph) < 1e-9 and abs(R - 1) < 1e-12
    ph2, _ = spike_phase([T_SPIKE * 0.999, T_SPIKE * 1.001])
    assert abs(ph2) < 0.01


def test_chain_output_bursts_at_the_beat_period():
    torch.set_num_threads(1)
    w, groups = build(N=48, chains=(("ACGT", (24.0, 24.0)),), warm=400)
    sig, ts = record(w, groups, 1400)
    st, h, env = spikes(sig[0], ts)
    assert len(st) >= 3
    assert np.allclose(np.diff(st), T_SPIKE, atol=12)
    assert env.max() > 2.0 * np.median(env)       # bursts above the typical level ...
    assert env.min() < 0.25 * env.max()           # ... with deep troughs between them
