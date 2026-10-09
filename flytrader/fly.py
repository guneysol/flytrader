"""The trader: market features -> fly senses -> brain -> behavior -> trade.

Innate reflexes (feed / escape / retreat) come from the whole-brain spiking
model. The mushroom body learns which market "smells" came before profits and
losses with the fly's own rules:

  * dopamine writes only into its own compartments (PAM -> reward, PPL1 ->
    punishment), read from DAN->MBON synapse counts in FlyWire;
  * learning is driven by reward *prediction error* (MBON output predicts the
    outcome; only the surprise gets written);
  * gamma-lobe compartments learn fast and forget fast, alpha/beta lobes slowly.

Decisions are slow evidence accumulation (Groschner et al. 2018); exits follow
patch-leaving foraging (marginal value theorem; Davidson & El Hady 2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuits import Circuits

ODOR_FEATURES = ["r1", "r5", "r15", "r60", "drawdown", "vol_z", "buy_ratio",
                 "buyers_z", "new_wallet_frac", "big_sell", "volatility", "trade_size_z",
                 "uniq_ratio", "buyers_accel"]

MAX_HZ = 200.0
LESIONS = ("none", "no_mushroom_body", "shuffled_wiring", "no_dopamine", "no_giant_fiber")


def clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


@dataclass
class Senses:
    sugar_hz: float
    bitter_hz: float
    loom_hz: float
    odor: np.ndarray  # PN rate per glomerulus


@dataclass
class Thought:
    """Everything the fly 'felt' and 'did' on one bar, for logging and charts."""
    senses: Senses
    mn9_hz: float
    gf_hz: float
    mdn_hz: float
    valence: float
    novelty: float
    hunger: float
    evidence: float
    kc_active: int
    action: str  # BUY / SELL / HOLD / WAIT
    reason: str
    kc_pattern: np.ndarray | None = None
    key: tuple = ()  # quantized (sugar, bitter, loom) drive: which brain sim this was


@dataclass
class Fly:
    c: Circuits
    seed: int = 7
    taste: str = "momentum"  # "momentum": rising price tastes sweet; "dip": oversold dips taste sweet
    lesion: str = "none"
    lr: float = 0.3  # dopamine plasticity strength
    sim_ms: float = 120.0
    entry_bound: float = 1.0  # accumulated evidence needed to buy
    leave_bound: float = 0.04  # accumulated "this patch is worse than elsewhere" needed to leave
    pain_sigmas: float = 2.5  # unrealized loss (in volatility units) that hurts enough to jump
    fee: float = 0.0125  # cost per side: the "travel cost" of moving to a new food patch
    record_ms: float | None = None  # keep per-frame spikes of each brain sim for the visualizer
    _cache: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        c = self.c
        n_g = len(c.glomeruli)
        # each market feature is an "odorant" with a sparse receptor profile over glomeruli,
        # separately for rising (+) and falling (-) values
        self.receptors = np.zeros((2 * len(ODOR_FEATURES), n_g))
        for i in range(len(self.receptors)):
            g = rng.choice(n_g, size=3, replace=False)
            self.receptors[i, g] = rng.uniform(0.3, 1.0, size=3)
        self.W_pn_kc = c.W_pn_kc.toarray()
        # KC excitability compensates for how much input a KC gets (Abdelrahman et al. 2021, PNAS),
        # otherwise a few hub KCs win for every odor and all smells look alike
        self.kc_gain = 1.0 / (self.W_pn_kc.sum(axis=1) + 1.0)
        self.odor_bg = None  # ORN adaptation state
        used = c.mbon_used.astype(np.float32)
        W0 = c.W_kc_mbon / (c.W_kc_mbon.sum(axis=0, keepdims=True) + 1e-9) * used
        self.approach = c.mbon_approach.copy()
        self.reward_gain, self.punish_gain = c.mbon_reward_gain.copy(), c.mbon_punish_gain.copy()
        if self.lesion == "shuffled_wiring":
            # same synapse statistics, but dopamine compartments and MBON valence assigned at random:
            # does the real reward/punishment wiring matter?
            perm = rng.permutation(len(self.approach))
            self.approach = self.approach[perm]
            self.reward_gain, self.punish_gain = self.reward_gain[perm], self.punish_gain[perm]
        self.W0 = W0
        self.W = W0.copy()
        # gamma lobe forgets in ~2h of bars, alpha/beta lobes in ~1-2 days
        self.recovery = (c.mbon_gamma * 1 / 120 + (1 - c.mbon_gamma) * 1 / 2000).astype(np.float32)
        self.novel_mbon = np.array([t in ("MBON16", "MBON17") for t in c.mbon_types])
        self.familiar = np.zeros(len(c.kc), dtype=np.float32)  # habituation per KC (alpha'3)
        self.trace: np.ndarray | None = None  # KC pattern at entry, eligible for dopamine
        self.hunger = 0.5
        self.ret_scale = 0.03  # running typical |trade return|, for reward prediction
        # dietary preference between momentum-sweet and dip-sweet; learned when taste == "auto"
        self.diet = {"momentum": [1.0, 0.0], "dip": [0.0, 1.0]}.get(self.taste, [0.5, 0.5])
        self._flavors = (0.0, 0.0)
        self.entry_flavors = (0.0, 0.0)
        self.evidence = 0.0
        self.leave = 0.0
        self.patch_rate = 0.0
        self.env_rate = 0.0
        self.peak = 0.0
        self.held = 0

    # ---- senses -------------------------------------------------------
    def sense(self, f: dict, in_position: bool, unrealized: float) -> Senses:
        h = self.hunger
        # tastes respond to *unusual* moves for this coin (z-scores), so one fly works on a calm
        # large-cap and on a coin that moves 10% a minute
        mom = (f["r5_z"] - 0.8) / 1.5 + 0.15 * f["buy_ratio_z"]
        dip = (-f["r15_z"] - 1.0) / 1.5 + 0.1 * max(f["big_sell_z"], 0)  # capitulation
        self._flavors = (clip01(mom), clip01(dip))
        sweet = 1.6 * (self.diet[0] * self._flavors[0] + self.diet[1] * self._flavors[1])
        sour = (-f["r5_z"] - 1.2) / 1.5 + (0.45 - f["buy_ratio"]) * 2 + 0.2 * max(f["drawdown_z"], 0)
        # hunger raises sugar gain first; only a starving fly ignores bitterness (Inagaki 2014)
        sugar = clip01(sweet) * MAX_HZ * (0.5 + h)
        bitter = clip01(sour) * MAX_HZ * (1.3 - 0.6 * max(h - 0.6, 0) / 0.4)
        # looming: a crash far outside this coin's normal minute, a >8% candle, or a whale dump
        loom = clip01(max((-f["r1_z"] - 2.5) / 2, (-f["r1"] - 0.08) / 0.1, (f["big_sell_z"] - 3) / 3))
        if in_position:
            # nociception: a loss larger than the market's own noise hurts, and pain triggers escape
            sigma = max(f["volatility"], 0.002) * np.sqrt(max(self.held, 1))
            pain = (-unrealized) / (self.pain_sigmas * sigma)
            trail = (self.peak - unrealized) / (2 * self.pain_sigmas * sigma) if self.peak > 0.01 else 0
            loom = max(loom, clip01(pain - 0.8), clip01(trail - 0.8))
        return Senses(min(sugar, MAX_HZ), min(bitter, MAX_HZ), loom * MAX_HZ, self._smell(f))

    def _smell(self, f: dict) -> np.ndarray:
        z = np.array([float(np.clip(f.get(n + "_z", f.get(n, 0.0)), -3, 3)) for n in ODOR_FEATURES])
        orn = np.concatenate([np.maximum(z, 0), np.maximum(-z, 0)]) @ self.receptors * 60
        # receptor adaptation: a smell that has been around for a while fades, changes stand out
        self.odor_bg = orn.copy() if self.odor_bg is None else 0.95 * self.odor_bg + 0.05 * orn
        orn = np.maximum(orn - 0.7 * self.odor_bg, 0)
        # antennal lobe divisive normalization (Olsen, Bhandawat & Wilson 2010)
        r15, sig, m = orn ** 1.5, 12 ** 1.5, (0.05 * orn.sum()) ** 1.5
        return 165 * r15 / (r15 + sig + m)

    # ---- brain --------------------------------------------------------
    def _motor(self, s: Senses) -> tuple[float, float, float]:
        q = lambda x: int(round(x / 10.0)) * 10  # noqa: E731 - quantize to 10 Hz so we can cache
        key = (q(s.sugar_hz), q(s.bitter_hz), q(s.loom_hz))
        if key == (0, 0, 0):
            return 0.0, 0.0, 0.0
        if key not in self._cache:
            r = self.c.brain.simulate(self.drive(key), t_ms=self.sim_ms, readout_from_ms=30,
                                      rng=np.random.default_rng(hash(key) & 0xFFFF), frame_ms=self.record_ms)
            if r.frames:
                self.frames[key] = r.frames
            self._cache[key] = (float(r.rates[self.c.mn9].mean()),
                                float(r.rates[self.c.gf].mean()),
                                float(r.rates[self.c.mdn].mean()))
        mn9, gf, mdn = self._cache[key]
        if self.lesion == "no_giant_fiber":
            gf = 0.0
        return mn9, gf, mdn

    def drive(self, key) -> dict[int, float]:
        drive = {}
        for pop, hz in zip((self.c.sugar, self.c.bitter, self.c.looming), key, strict=True):
            if hz:
                drive.update({int(i): float(hz) for i in pop})
        return drive

    def _kc(self, odor: np.ndarray) -> np.ndarray:
        """Kenyon cells with APL feedback inhibition: threshold rises with total KC activity."""
        x = (self.W_pn_kc @ odor) * self.kc_gain
        if x.max() <= 0:
            return np.zeros(len(x), dtype=bool)
        theta = np.quantile(x, 0.90)
        for _ in range(4):
            active = x > theta
            frac = active.mean()
            theta *= 1 + 2.0 * (frac - 0.06)  # APL pushes sparseness toward ~5-10%
        return x > theta

    def _mbon(self, kc: np.ndarray) -> np.ndarray:
        return self.W[kc].sum(axis=0)

    def _valence(self, kc: np.ndarray) -> float:
        if self.lesion == "no_mushroom_body" or not kc.any():
            return 0.0

        def ratio(W):
            out = W[kc].sum(axis=0)
            app, avo = out[self.approach].sum(), out[~self.approach].sum()
            return (app - avo) / (app + avo + 1e-9)
        # learned change vs. the naive fly: an unfamiliar smell starts neutral
        return float(np.tanh(6 * (ratio(self.W) - ratio(self.W0))))

    def _novelty(self, kc: np.ndarray) -> float:
        if not kc.any():
            return 0.0
        nov = 1.0 - float(self.familiar[kc].mean())
        self.familiar[kc] += 0.05 * (1 - self.familiar[kc])  # alpha'3 habituation (Hattori 2017)
        self.familiar *= 0.9995
        return nov

    # ---- behavior -----------------------------------------------------
    def think(self, f: dict, in_position: bool, unrealized: float = 0.0, bar_ret: float = 0.0) -> Thought:
        if in_position:
            self.held += 1
            self.peak = max(self.peak, unrealized)
            self.patch_rate = 0.7 * self.patch_rate + 0.3 * bar_ret
        else:
            self.hunger = min(1.0, self.hunger + 1 / 240)  # gets hungrier the longer it doesn't eat
        self.env_rate = 0.98 * self.env_rate + 0.02 * float(np.clip(f["r1"], -0.05, 0.05))
        # forgetting: every minute synapses drift back toward their naive weights
        self.W += self.recovery * (self.W0 - self.W)
        if self.taste == "auto":
            self.diet = [d + (0.5 - d) / 500 for d in self.diet]

        s = self.sense(f, in_position, unrealized)
        mn9, gf, mdn = self._motor(s)
        q = lambda x: int(round(x / 10.0)) * 10  # noqa: E731
        key = (q(s.sugar_hz), q(s.bitter_hz), q(s.loom_hz))
        kc = self._kc(s.odor)
        val = self._valence(kc)
        nov = self._novelty(kc)

        if not in_position:
            # leaky evidence accumulation; an unfamiliar market needs more evidence
            # marginal value theorem: a patch is only worth flying to if the expected gain beats the
            # travel cost. Expected gain ~ how much this market moves over a typical 15-minute meal.
            expected_move = max(f["volatility"], 1e-4) * np.sqrt(15)
            travel = min(2 * self.fee / expected_move, 3.0)
            push = mn9 / 50.0 + 0.8 * val - 0.4 - 0.3 * nov - 0.5 * travel
            self.evidence = 0.6 * self.evidence + push if mn9 > 0 else 0.5 * self.evidence
            self.evidence = max(self.evidence, -1.0)
            if gf > 5 or mdn > 5:
                action, why = "WAIT", "startled"
                self.evidence = 0.0
            elif self.evidence > self.entry_bound:
                action, why = "BUY", f"proboscis out ({mn9:.0f} Hz), smell {val:+.2f}, hunger {self.hunger:.2f}"
            else:
                action, why = "WAIT", ("not hungry" if mn9 == 0 else f"tasting… ({self.evidence:.2f})")
        else:
            # patch leaving: accumulate how much worse this patch is than the environment
            self.leave = max(0.0, 0.8 * self.leave + (self.env_rate - self.patch_rate) - 0.3 * val * 0.01)
            if gf > 5:
                action, why = "SELL", f"giant fiber escape ({gf:.0f} Hz)"
            elif mdn > 5:
                action, why = "SELL", f"moonwalked away ({mdn:.0f} Hz)"
            elif mn9 == 0 and s.bitter_hz > s.sugar_hz + 60 * (1 + val):
                action, why = "SELL", "tastes bitter, spat it out"
            elif self.leave > self.leave_bound and self.held >= 3:
                action, why = "SELL", "patch depleted, flew off"
            else:
                action, why = "HOLD", "still eating" if mn9 > 0 else "sitting on it"

        if action == "BUY":
            self.trace = kc.copy()
            self.entry_flavors = self._flavors
            self.held, self.peak, self.leave, self.patch_rate = 0, 0.0, 0.0, 0.0
            self.evidence = 0.0
        return Thought(s, mn9, gf, mdn, val, nov, self.hunger, self.evidence, int(kc.sum()), action, why, kc, key)

    def dopamine(self, ret: float):
        """Learn from the outcome of a trade with a reward-prediction-error rule.

        The KCs active at entry predicted value v̂ via MBON output; dopamine
        carries δ = ret − v̂. PAM (δ>0) depresses KC>avoid-MBON synapses, PPL1
        (δ<0) depresses KC>approach-MBON synapses, each only in compartments it
        actually innervates. Without dopamine, synapses drift back (forgetting).
        """
        if self.trace is None or self.lesion in ("no_dopamine", "no_mushroom_body"):
            return
        # the MBON output predicts the outcome in units of this market's typical trade result
        predicted = self._valence(self.trace) * self.ret_scale
        delta = ret - predicted
        self.ret_scale = 0.9 * self.ret_scale + 0.1 * max(abs(ret), 0.005)
        strength = min(abs(delta) / (2 * self.ret_scale), 1.0) * self.lr
        gain = self.reward_gain if delta > 0 else self.punish_gain
        # gamma compartments write quickly, alpha/beta slowly (Aso & Rubin 2016)
        speed = 0.3 + 0.7 * self.c.mbon_gamma
        self.W[self.trace] *= (1.0 - strength * gain * speed)

    def end_trade(self, ret: float):
        self.dopamine(ret)
        if self.taste == "auto" and sum(self.entry_flavors) > 0:
            # dietary plasticity: sweet sensitivity shifts toward the flavor that paid off
            share = np.array(self.entry_flavors) / sum(self.entry_flavors)
            d = np.array(self.diet) + 0.1 * np.tanh(ret / (2 * self.fee + 0.005)) * share
            d = np.clip(d, 0.1, 1.0)
            self.diet = list(d / d.sum())
        self.trace = None
        # a good meal satisfies; a bad one leaves it hungrier
        if ret > 0:
            self.hunger = clip01(self.hunger - 0.6 * clip01(ret / 0.05))
        else:
            self.hunger = min(1.0, self.hunger + 0.05)
        self.leave = 0.0
