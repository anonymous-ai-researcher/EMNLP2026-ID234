"""Lookups across distance (Section 5.5; Figure 4; Appendix F.7, Tables 41-42).

Each architecture is tuned once on development accuracy at distances 0-4 (grid:
learning rate {3e-4, 1e-3, 3e-3}, weight decay {0, 0.1, 1}, {50,000, 200,000}
steps, Mamba state size {16, 64, 256}); the selected settings (``TUNED``) are then
used for every width.  Widths 32 ... 2048 (23 values; Mamba additionally 2176 ...
3072 at d = 10 and 11), five seeds, 135-token vocabulary, context 64, batch 64,
500 warm-up steps, cosine schedule, no dropout.  Each test distance 0..16 has
2,000 items.  The Mamba state probe reads the second block's selective-scan state
(width 128, state sizes 16/64/256) after each of the 17 tokens following the
subject.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List

import numpy as np
import torch

from ..alignment import probes as PROBES
from ..analysis import scaling as SC
from ..data import datasets as D
from ..data import lexicon as lx
from ..data import templates as T
from ..data.vocab import vocab_for
from ..models.architectures import parameter_formula
from ..models.training import MODEL_SEEDS, TUNED, TrainConfig, accuracy, encode_batch, train
from .common import FULL, Scale, run_dir, save

WIDTHS = (32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
          1280, 1536, 1792, 2048)
MAMBA_EXTRA = (2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072)
TUNING_GRID = {"lr": (3e-4, 1e-3, 3e-3), "weight_decay": (0.0, 0.1, 1.0), "steps": (50_000, 200_000),
               "d_state": (16, 64, 256)}


def tune(arch: str, scale: Scale = FULL, width: int = 128) -> Dict:
    data = D.distance_dataset(); vocab = vocab_for("distance")
    best = None
    states = TUNING_GRID["d_state"] if arch == "mamba" else (16,)
    for lr, wd, steps, ds in itertools.product(TUNING_GRID["lr"], TUNING_GRID["weight_decay"],
                                               TUNING_GRID["steps"], states):
        cfg = TrainConfig(arch=arch, d=width, max_len=64, lr=lr, weight_decay=wd, steps=scale.s(steps),
                          d_state=ds, seed=MODEL_SEEDS[0], checkpoint_every=scale.s(steps))
        m = train(data["train"], vocab, cfg)
        acc = accuracy(m, data["val"], vocab, cfg.device)
        if best is None or acc > best[0]:
            best = (acc, dict(lr=lr, weight_decay=wd, steps=steps, d_state=ds))
    save(f"tuning_{arch}", {"dev_accuracy": best[0], "selected": best[1]})
    return best[1]


def sweep(arch: str, scale: Scale = FULL, widths=WIDTHS) -> Dict:
    data = D.distance_dataset(); vocab = vocab_for("distance")
    hp = dict(TUNED[arch])
    res: Dict = {"params": {}, "accuracy": {}}
    ws = list(widths) + (list(MAMBA_EXTRA) if arch == "mamba" else [])
    for w in ws:
        res["params"][w] = parameter_formula(arch, w)
        for seed in MODEL_SEEDS:
            cfg = TrainConfig(arch=arch, d=w, max_len=64, seed=seed, lr=hp["lr"], weight_decay=hp["weight_decay"],
                              steps=scale.s(hp["steps"]), d_state=hp.get("d_state", 16),
                              checkpoint_every=scale.s(hp["steps"]))
            m = train(data["train"], vocab, cfg, run_dir("arch", arch, w, seed))
            for d, items in data["test"].items():
                if w in MAMBA_EXTRA and d not in (10, 11):
                    continue
                res["accuracy"].setdefault(f"{w}/{d}", []).append(accuracy(m, items, vocab, cfg.device))
    save(f"arch_sweep_{arch}", res)
    return res


def growth(arch: str) -> Dict:
    from .common import load
    r = load(f"arch_sweep_{arch}")
    ds = list(range(17))
    grid = np.array([[[r["accuracy"][f"{w}/{d}"][s] for w in WIDTHS] for d in ds] for s in range(5)])
    params = [parameter_formula(arch, w) for w in WIDTHS]
    out = {"main": SC.growth_with_uncertainty(grid, params, ds),
           "log_interpolation": SC.growth_with_uncertainty(grid, params, ds, interpolate=True),
           "exact_enumeration": SC.exact_enumeration(grid, params, ds)}
    save(f"growth_{arch}", out)
    return out


def mamba_state_probe(scale: Scale = FULL, offsets=range(1, 18), sizes=(4_096, 1_024, 512)) -> Dict:
    """Linear probe for the subject's number in the second block's state, per offset and
    per condition (intervening nouns with the same or the opposite number)."""
    data = D.distance_dataset(); vocab = vocab_for("distance")
    res: Dict = {}
    for d_state in (16, 64, 256):
        cfg = TrainConfig(arch="mamba", d=128, max_len=64, d_state=d_state, seed=MODEL_SEEDS[0],
                          lr=1e-3, weight_decay=0.0, steps=scale.s(200_000), checkpoint_every=scale.s(200_000))
        m = train(data["train"], vocab, cfg, run_dir("mamba_probe", d_state))
        mixer = m.blocks[1].mixer
        for cond in ("same", "opposite"):
            for t in offsets:
                X, y = {}, {}
                for split, n in zip(("train", "dev", "test"), sizes):
                    rng = random.Random(f"probe{d_state}{cond}{t}{split}")
                    sents = []
                    for _ in range(scale.n(n)):
                        sn = rng.randrange(2)
                        f = T.agreement(rng, 6)
                        w = list(f.good.words)
                        w[1] = lx.ANIMATE_NOUNS[rng.randrange(20)][sn]
                        for i in range(2, len(w)):
                            if w[i] in {p for q in lx.PLACE_NOUNS for p in q}:
                                pair = next(q for q in lx.PLACE_NOUNS if w[i] in q)
                                w[i] = pair[sn if cond == "same" else 1 - sn]
                        f.good.words = w[:min(len(w), 2 + t)]; f.good.target = len(f.good.words) - 1
                        sents.append((f.good, sn))
                    ids, tgt, _ = encode_batch([s for s, _ in sents], vocab, cfg.device)
                    mixer.record = True
                    with torch.no_grad():
                        m.encode(ids, "block2")
                    st = mixer.last_states[torch.arange(len(sents)), tgt].flatten(1).cpu().numpy()
                    mixer.record = False
                    X[split], y[split] = st, [sn for _, sn in sents]
                clf, sd = PROBES.fit_probe(X["train"], y["train"])
                res[f"{d_state}/{cond}/{t}"] = PROBES.probe_accuracy(clf, sd, X["test"], y["test"])
    save("mamba_state_probe", res)
    return res
