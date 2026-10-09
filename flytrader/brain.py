"""Whole-brain leaky integrate-and-fire model of the adult fruit fly.

Port of Shiu et al. 2024 ("A Drosophila computational brain model reveals
sensorimotor processing", Nature) from Brian2 to plain numpy/scipy so a single
decision (~150 ms of simulated brain time) runs in well under a second.

Wiring: FlyWire v783 connectome as packaged by Shiu et al.: 138,639 neurons, ~15M connections.
Same constants as the paper: every synapse adds 0.275 mV (signed by the
presynaptic neurotransmitter) to an exponentially decaying conductance after 1.8 ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import paths

DATA = paths.DATA / "connectome"

# Shiu et al. constants (ms, mV)
V_0, V_RST, V_TH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, F_POI = 0.275, 250


@dataclass
class SimResult:
    rates: np.ndarray  # Hz per neuron over the readout window
    spikes_total: int
    sim_ms: float
    frames: list[np.ndarray] | None = None  # spiking neuron indices per frame (if recorded)


@dataclass
class Brain:
    ids: np.ndarray  # FlyWire root ids, index-aligned with the weight matrix
    W: sp.csc_matrix  # post x pre, mV per spike
    index: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Brain:
        if not (DATA / "Completeness_783.csv").exists():
            raise SystemExit(f"connectome data not found in {DATA}; run: python scripts/setup_data.py")
        cache = DATA / "brain_cache.npz"
        ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index.values.astype(np.int64)
        if cache.exists():
            W = sp.load_npz(cache).tocsc()
        else:
            con = pd.read_parquet(
                DATA / "Connectivity_783.parquet",
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
            )
            n = len(ids)
            W = sp.csc_matrix(
                (
                    con["Excitatory x Connectivity"].values.astype(np.float32) * W_SYN,
                    (con["Postsynaptic_Index"].values, con["Presynaptic_Index"].values),
                ),
                shape=(n, n),
            )
            sp.save_npz(cache, W)
        return cls(ids=ids, W=W, index={int(r): i for i, r in enumerate(ids)})

    @property
    def n(self) -> int:
        return len(self.ids)

    def idx(self, root_ids) -> np.ndarray:
        return np.array([self.index[int(r)] for r in root_ids if int(r) in self.index], dtype=np.int64)

    def simulate(
        self,
        drive: dict[int, float],
        t_ms: float = 150.0,
        readout_from_ms: float = 50.0,
        dt: float = 0.1,
        rng: np.random.Generator | None = None,
        frame_ms: float | None = None,
    ) -> SimResult:
        """Run the brain with Poisson drive on some neurons.

        drive maps neuron index -> Poisson rate in Hz (the paper's way of
        "activating" sensory neurons). Returns per-neuron firing rates
        measured from readout_from_ms to t_ms.
        """
        rng = rng or np.random.default_rng()
        n = self.n
        v = np.full(n, V_0, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        rfc_until = np.full(n, -1.0, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)

        steps = int(round(t_ms / dt))
        delay = int(round(T_DLY / dt)) - 1  # ring of delay+1 slots = 1.8 ms
        ring: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * (delay + 1)

        d_idx = np.fromiter(drive.keys(), dtype=np.int64, count=len(drive))
        d_p = np.fromiter(drive.values(), dtype=np.float64, count=len(drive)) * dt / 1000.0
        # Poisson-driven neurons have no refractory period, as in the paper
        no_rfc = np.zeros(n, dtype=bool)
        no_rfc[d_idx] = True

        decay_g = np.float32(np.exp(-dt / TAU))
        kick = np.float32(W_SYN * F_POI)
        readout_step = int(round(readout_from_ms / dt))
        total = 0
        per_frame = int(round(frame_ms / dt)) if frame_ms else 0
        frames: list[list[np.ndarray]] = [[] for _ in range(-(-steps // per_frame))] if per_frame else []

        for s in range(steps):
            t = s * dt
            active = rfc_until <= t
            # dv/dt = (v0 - v + g) / t_mbr, dg/dt = -g / tau (forward Euler, dt = 0.1 ms);
            # frozen while refractory, like Brian2's "unless refractory"
            dv = (V_0 - v + g) * (dt / T_MBR)
            v = np.where(active, v + dv, v)
            g = np.where(active, g * decay_g, g)

            # delayed synaptic input from spikes `delay` steps ago
            arrived = ring[s % (delay + 1)]
            if arrived.size:
                g += np.asarray(self.W[:, arrived].sum(axis=1)).ravel()

            # external Poisson drive
            if d_idx.size:
                hits = rng.random(d_idx.size) < d_p
                if hits.any():
                    g[d_idx[hits]] += kick

            spiked = np.flatnonzero((v > V_TH) & active)
            if spiked.size:
                v[spiked] = V_RST
                g[spiked] = 0.0
                rfc_until[spiked] = np.where(no_rfc[spiked], t, t + T_RFC)
                if s >= readout_step:
                    counts[spiked] += 1
                total += spiked.size
                if per_frame:
                    frames[s // per_frame].append(spiked)
            ring[s % (delay + 1)] = spiked

        window_s = (t_ms - readout_from_ms) / 1000.0
        out = [np.unique(np.concatenate(f)).astype(np.uint32) if f else np.empty(0, np.uint32) for f in frames]
        return SimResult(rates=counts / window_s, spikes_total=total, sim_ms=t_ms, frames=out or None)
