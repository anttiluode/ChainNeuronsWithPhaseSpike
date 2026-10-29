"""The near-complement pair is excitable: threshold, all-or-none slip, deep-lock control is graded."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.theta_neuron import drive, slips, world


def test_near_complement_pair_fires_all_or_none():
    w = world(0.015, [(24.0, 24.0)], warm=2000)
    sub = drive(w, (24.0, 26.5), [(100, 0.03)], 1800)[:, 0]
    sup1 = drive(w, (24.0, 26.5), [(100, 0.15)], 1800)[:, 0]
    sup2 = drive(w, (24.0, 26.5), [(100, 0.40)], 1800)[:, 0]
    assert slips(sub[-1]) == 0 and abs(sub).max() < 1.5
    assert slips(sup1[-1]) == 1 and slips(sup2[-1]) == 1
    assert abs(abs(sup1).max() - abs(sup2).max()) < 0.2       # same size regardless of input strength


def test_exact_complement_is_graded_not_firing():
    w = world(0.0, [(24.0, 24.0)], warm=2000)
    s = drive(w, (24.0, 26.5), [(100, 0.15)], 1800)[:, 0]
    assert slips(s[-1]) == 0 and abs(s).max() < 2.0
