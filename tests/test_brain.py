"""Reproduces key results of Shiu et al. 2024 with our numpy port. Needs scripts/setup_data.py."""

import numpy as np
import pytest

from flytrader.brain import DATA

pytestmark = pytest.mark.data
if not (DATA / "Connectivity_783.parquet").exists():
    pytest.skip("connectome not downloaded (python scripts/setup_data.py)", allow_module_level=True)


@pytest.fixture(scope="module")
def c():
    from flytrader.circuits import Circuits
    return Circuits.load()


def rates(c, pops, hz=150, t_ms=200):
    drive = {int(i): hz for p in pops for i in p}
    return c.brain.simulate(drive, t_ms=t_ms, readout_from_ms=50, rng=np.random.default_rng(1)).rates


def test_sugar_drives_proboscis_motor_neuron(c):
    assert rates(c, [c.sugar])[c.mn9].mean() > 20


def test_bitter_alone_does_not_and_suppresses_sugar(c):
    assert rates(c, [c.bitter])[c.mn9].mean() == 0
    assert rates(c, [c.sugar, c.bitter])[c.mn9].mean() < rates(c, [c.sugar])[c.mn9].mean()


def test_looming_triggers_giant_fiber_escape(c):
    assert rates(c, [c.looming], hz=100)[c.gf].mean() > 50


def test_dopamine_compartments_match_biology(c):
    # reward (PAM) innervates avoidance MBONs, punishment (PPL1) innervates approach MBONs (Aso et al. 2014)
    used = c.mbon_used
    assert (c.mbon_reward_gain[used & ~c.mbon_approach] > c.mbon_punish_gain[used & ~c.mbon_approach]).all()
    types = {t for t, a, u in zip(c.mbon_types, c.mbon_approach, used, strict=True) if a and u}
    assert {"MBON11", "MBON12", "MBON13", "MBON14"} <= types
