"""Named neuron populations in the FlyWire connectome that the trader uses.

Senses (inputs)
  sugar GRNs    -> "this tastes good"        <- buy pressure / upward momentum
  bitter GRNs   -> "this tastes bad"         <- sell pressure / drawdown
  LC4 + LPLC2   -> looming object detectors  <- a crash coming at you fast
  uPNs          -> smell (antennal lobe out) <- the overall "scent" of the market

Behaviors (outputs)
  MN9           -> proboscis extension (eat)  => BUY
  DNp01 (GF)    -> giant fiber escape jump    => PANIC SELL
  MDN           -> moonwalk backwards         => back off / SELL

Learning
  Kenyon cells -> MBONs, with dopamine (PAM = reward, PPL1 = punishment)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .brain import DATA, Brain

# Shiu et al. 2024 (v630 ids), right-hemisphere labellar GRNs. One of each list no longer exists in
# v783 and is skipped by Brain.idx, leaving 20 sugar + 20 bitter neurons.
SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
BITTER = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
    720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
    720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
    720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
    720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
# MN9 pair (cell type CB0701 in FlyWire v783)
MN9 = [720575940660219265, 720575940618238523]


@dataclass
class Circuits:
    brain: Brain
    sugar: np.ndarray
    bitter: np.ndarray
    looming: np.ndarray
    mn9: np.ndarray
    gf: np.ndarray
    mdn: np.ndarray
    glomeruli: list[str]
    pn_by_glom: list[np.ndarray]
    kc: np.ndarray
    mbon: np.ndarray
    mbon_types: list[str]
    mbon_approach: np.ndarray  # bool over mbon: PPL1-innervated (punishment depresses) = approach
    mbon_reward_gain: np.ndarray  # PAM -> MBON synapses, normalized: how hard reward writes here
    mbon_punish_gain: np.ndarray  # PPL1 -> MBON synapses, normalized
    mbon_gamma: np.ndarray  # share of KC input from gamma lobe KCs (fast, short-lived memories)
    mbon_used: np.ndarray  # typical MBONs with real KC input and dopamine innervation
    W_pn_kc: sp.csr_matrix  # kc x glomerulus, synapse counts
    W_kc_mbon: np.ndarray  # kc x mbon, synapse counts
    dan_pam: np.ndarray
    dan_ppl1: np.ndarray

    @classmethod
    def load(cls, brain: Brain | None = None) -> Circuits:
        brain = brain or Brain.load()
        a = pd.read_csv(DATA / "annotations.tsv", sep="\t", low_memory=False)
        ct = a.cell_type.fillna("")

        def pick(mask) -> np.ndarray:
            return brain.idx(a.loc[mask, "root_id"])

        # uniglomerular projection neurons, grouped by glomerulus (e.g. "DM1_lPN" -> DM1)
        upn = a[(a.cell_class == "ALPN") & (a.cell_sub_class == "uniglomerular")].copy()
        upn["glom"] = upn.cell_type.str.split("_").str[0]
        groups = {g: brain.idx(df.root_id) for g, df in upn.groupby("glom")}
        groups = {g: v for g, v in groups.items() if len(v)}
        glomeruli = sorted(groups)

        kc = pick(a.cell_class == "Kenyon_Cell")
        mbon_df = a[a.cell_class == "MBON"].copy()
        mbon_df["i"] = mbon_df.root_id.map(lambda r: brain.index.get(int(r), -1))
        mbon_df = mbon_df[mbon_df.i >= 0]
        mbon = mbon_df.i.values.astype(np.int64)

        W = brain.W  # post x pre, mV; recover synapse counts
        counts = abs(W) / 0.275
        pn_cols = []
        for g in glomeruli:
            pn_cols.append(np.asarray(counts[kc][:, groups[g]].sum(axis=1)).ravel())
        W_pn_kc = sp.csr_matrix(np.stack(pn_cols, axis=1))
        W_kc_mbon = counts[mbon][:, kc].T.toarray().astype(np.float32)

        # dopamine compartments, read straight from DAN -> MBON synapse counts
        dt = a.cell_type.fillna("")
        pam, ppl1 = pick(dt.str.startswith("PAM")), pick(dt.str.match(r"PPL1\d"))
        from_pam = np.asarray(counts[mbon][:, pam].sum(axis=1)).ravel()
        from_ppl1 = np.asarray(counts[mbon][:, ppl1].sum(axis=1)).ravel()
        kc_types = a.set_index("root_id").reindex(brain.ids[kc]).cell_sub_class.fillna("").values
        gamma = W_kc_mbon[kc_types == "KCg"].sum(axis=0) / (W_kc_mbon.sum(axis=0) + 1e-9)
        kc_in = W_kc_mbon.sum(axis=0)
        used = (kc_in > 200) & ((from_pam + from_ppl1) > 100)

        return cls(
            brain=brain,
            sugar=brain.idx(SUGAR),
            bitter=brain.idx(BITTER),
            looming=pick(ct.isin(["LC4", "LPLC2"])),
            mn9=brain.idx(MN9),
            gf=pick(ct == "DNp01"),
            mdn=pick(ct == "MDN"),
            glomeruli=glomeruli,
            pn_by_glom=[groups[g] for g in glomeruli],
            kc=kc,
            mbon=mbon,
            mbon_types=mbon_df.cell_type.tolist(),
            mbon_approach=from_ppl1 > from_pam,
            mbon_reward_gain=from_pam / from_pam.max(),
            mbon_punish_gain=from_ppl1 / from_ppl1.max(),
            mbon_gamma=gamma,
            mbon_used=used,
            W_pn_kc=W_pn_kc,
            W_kc_mbon=W_kc_mbon,
            dan_pam=pam,
            dan_ppl1=ppl1,
        )
