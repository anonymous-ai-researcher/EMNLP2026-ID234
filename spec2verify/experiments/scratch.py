"""From-scratch experiments: Table 2 (2L models), Figures 2-5, Appendices D, E, F.

Every function takes a ``Scale`` (``common.FULL`` reproduces the protocol) and
writes its results with ``common.save``.  Models are cached under ``runs/``.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..alignment import das as DAS
from ..alignment import joint as JOINT
from ..alignment import probes as PROBES
from ..alignment import unconstrained as UNC
from ..alignment.backbone import ScratchBackbone
from ..analysis import crossing as CROSS
from ..analysis import stats as ST
from ..brasp import programs as P
from ..brasp.scm import compile_program
from ..data import datasets as D
from ..data import pairs as PR
from ..data import templates as T
from ..data.vocab import vocab_for
from ..models.architectures import build_model, site_for
from ..models.training import MODEL_SEEDS, TrainConfig, load_model, train
from ..sae import autoencoders as SAE
from ..sae import interchange as SX
from .common import FULL, Scale, run_dir, save

PHENOMENA = ("agreement", "npi", "binding", "concord")
ARCHS = ("transformer", "lstm", "mamba")
FROM_SCRATCH_DAS_SEEDS = DAS.DAS_SEEDS[:3]
DAS_PAIRS_PER_STRATUM = (3_000, 1_000, 1_000)


# ----------------------------------------------------------------------------- data and models
def core_data(ph: str, scale: Scale = FULL):
    sizes = tuple(scale.n(x) for x in D.CORE_SIZES[ph])
    return D.core_dataset(ph, sizes)


def config(arch: str, seed: int, scale: Scale = FULL, **kw) -> TrainConfig:
    cfg = TrainConfig(arch=arch, seed=seed, **kw)
    cfg.steps = scale.s(cfg.steps)
    cfg.checkpoint_every = min(cfg.checkpoint_every, cfg.steps)
    return cfg


def get_model(tag: str, train_set, vocab, cfg: TrainConfig, eval_sets=None, **kw):
    """Train once and cache the final checkpoint (and the per-checkpoint files)."""
    out = run_dir(tag, cfg.arch, cfg.seed)
    final = os.path.join(out, f"step{cfg.steps:06d}.pt")
    if os.path.exists(final):
        m = build_model(cfg.arch, len(vocab), cfg.d, cfg.max_len, cfg.d_state)
        m.load_state_dict(torch.load(final, map_location="cpu"))
        m.training_log = torch.load(os.path.join(out, "run.pt"))["log"]
        return m.to(cfg.device).eval(), out
    return train(train_set, vocab, cfg, out, eval_sets, **kw), out


def checkpoint(out: str, cfg: TrainConfig, vocab, step: int):
    m = build_model(cfg.arch, len(vocab), cfg.d, cfg.max_len, cfg.d_state)
    m.load_state_dict(torch.load(os.path.join(out, f"step{step:06d}.pt"), map_location="cpu"))
    return m.to(cfg.device).eval()


#: Share of DAS pairs whose source leaves the variable (and so the output) unchanged, so that
#: an ineffective intervention scores 0.5 (Appendix F.1; see README).
UNCHANGED_FRACTION = 0.5


def das_pairs(ph: str, data, variable: str, scale: Scale, seed: int = 42, exclude=None,
              dataset: Optional[str] = None, scm=None, **kw) -> Dict[str, List[PR.Pair]]:
    """Fresh DAS families (disjoint from the training families), per structural-load level:
    >= 5,000 pairs per level, split 60/20/20 by family (3,000/1,000/1,000)."""
    kw.setdefault("unchanged_fraction", UNCHANGED_FRACTION)
    exclude = exclude if exclude is not None else {s.family for s in data["train"]}
    dataset = dataset or ph
    levels = T.LOAD_LEVELS.get(ph, [0])
    out = {"train": [], "dev": [], "test": []}
    for load in levels:
        rng = D.stream(seed, "das", dataset, variable, load)
        gen = T.CORE_GENERATORS[ph]
        need = scale.n(sum(DAS_PAIRS_PER_STRATUM))
        fams = []
        while len(fams) < need:
            f = gen(rng, load)
            if f.good.family not in exclude:
                fams.append(f)
        sents = D.to_sentences(fams)
        pairs = PR.make_pairs(sents, variable, rng, scm or PR.scm_for(dataset), twins=PR.twins_of(sents), **kw)
        sp = PR.split_by_family(pairs, rng)
        for k, q in zip(("train", "dev", "test"), DAS_PAIRS_PER_STRATUM):
            out[k] += sp[k][:scale.n(q)]
    return out


# ----------------------------------------------------------------------------- Table 2, Table 18 (controls)
def table2(arch: str = "transformer", scale: Scale = FULL, seeds: Optional[Sequence[int]] = None) -> Dict:
    """Per-variable IIA of the 2-layer models (fixed sites, rank searched), the three
    negative controls, and the variance decomposition of Appendix C.3."""
    seeds = seeds or MODEL_SEEDS
    res: Dict = {"arch": arch, "variables": {}, "controls": {}, "variance_share": {}}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        cfg = config(arch, MODEL_SEEDS[0], scale)
        for seed in seeds:
            cfg = config(arch, seed, scale)
            model, out = get_model(f"core/{ph}", data["train"], vocab, cfg,
                                   eval_sets={"train": data["train"][:5000], "test": data["test"]})
            bb = ScratchBackbone(model, vocab)
            rand_bb = ScratchBackbone(checkpoint(out, cfg, vocab, 0), vocab)
            for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
                pairs = das_pairs(ph, data, var, scale)
                site = site_for(model, kind)
                runs = [DAS.search(bb, pairs, s, sites=[site], epochs=scale.e(100)) for s in FROM_SCRATCH_DAS_SEEDS]
                res["variables"].setdefault(f"{ph}/{var}", {})[seed] = [r.test_iia for r in runs]
                best = runs[0]
                ctrl = res["controls"].setdefault(f"{ph}/{var}", {})
                ctrl.setdefault("random_init", {})[seed] = DAS.refit_fixed(
                    rand_bb, pairs, site, best.rank, 701, epochs=scale.e(100)).test_iia
                ctrl.setdefault("shuffled", {})[seed] = DAS.refit_fixed(
                    bb, pairs, site, best.rank, 701, shuffle_labels=True, epochs=scale.e(100)).test_iia
                for other in PHENOMENA:
                    if other == ph:
                        continue
                    wrong = compile_program(P.core_program(other))
                    wvar = PR.VARIABLES[other][0 if kind == "atomic" else 1]
                    relabeled = {k: [PR.Pair(p.base, p.source, wvar, PR.label_pair(wrong, p.base, p.source, wvar)[0],
                                             p.base_output, p.family, p.load) for p in v] for k, v in pairs.items()}
                    ctrl.setdefault(f"wrong_scm:{other}", {})[seed] = DAS.refit_fixed(
                        bb, relabeled, site, best.rank, 701, epochs=scale.e(100)).test_iia
    for key, by_seed in res["variables"].items():
        arr = np.array([v for v in by_seed.values()])
        res["variance_share"][key] = ST.variance_decomposition(arr) if arr.shape[0] > 1 else None
    save(f"table2_scratch_{arch}", res)
    return res


# ----------------------------------------------------------------------------- D.1 rank ablation
def rank_ablation(scale: Scale = FULL, ranks=(0, 1, 2, 4, 8, 128)) -> Dict:
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config("transformer", seed, scale))
            bb = ScratchBackbone(model, vocab)
            for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
                pairs = das_pairs(ph, data, var, scale); site = site_for(model, kind)
                for r in ranks:
                    if r == 0:
                        y = torch.tensor([p.label for p in pairs["test"]])
                        iia = (bb.logits([p.base for p in pairs["test"]]).argmax(-1).cpu() == y).float().mean().item()
                    elif r >= bb.d:
                        iia = DAS.evaluate(bb, pairs["test"], site, torch.eye(bb.d), bb.d)
                    else:
                        iia = DAS.refit_fixed(bb, pairs, site, r, 701, epochs=scale.e(100)).test_iia
                    res.setdefault(f"{ph}/{var}", {}).setdefault(r, []).append(iia)
    save("rank_ablation", res)
    return res


def rank_by_layer(scale: Scale = FULL) -> Dict:
    """Appendix F.6: compositional IIA at every site and rank, on pairs in which exactly one
    input changes (two-input variables; the three-input NPI variable is in ``three_input``)."""
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config("transformer", seed, scale))
            bb = ScratchBackbone(model, vocab)
            pairs = das_pairs(ph, data, PR.VARIABLES[ph][1], scale)
            for site in ("block1", "block2_attn", "block2"):
                for r in (1, 2, 4, 8, 128):
                    iia = (DAS.evaluate(bb, pairs["test"], site, torch.eye(bb.d), bb.d) if r == 128 else
                           DAS.refit_fixed(bb, pairs, site, r, 701, epochs=scale.e(100)).test_iia)
                    res.setdefault(site, {}).setdefault(r, []).append(iia)
    save("rank_by_layer", res)
    return res


def three_input(scale: Scale = FULL) -> Dict:
    """Appendix D.1: the three-input NPI variable needs rank 3 before the last MLP."""
    sizes = tuple(scale.n(x) for x in (40_000, 4_000, 8_000))
    data = D.npi3_dataset(sizes); vocab = vocab_for("npi3")
    res: Dict = {}
    scm = PR.scm_for("npi3")
    for seed in MODEL_SEEDS:
        cfg = config("transformer", seed, scale, max_len=64)
        model, _ = get_model("npi3", data["train"], vocab, cfg)
        bb = ScratchBackbone(model, vocab)
        rng = random.Random(seed)
        pairs = PR.split_by_family(PR.make_pairs(data["test"] + data["val"], "npi_ok3", rng, scm), rng)
        for r in (0, 1, 2, 3, 4, 8, 128):
            if r == 0:
                y = torch.tensor([p.label for p in pairs["test"]])
                iia = (bb.logits([p.base for p in pairs["test"]]).argmax(-1).cpu() == y).float().mean().item()
            elif r == 128:
                iia = DAS.evaluate(bb, pairs["test"], "block2_attn", torch.eye(bb.d), bb.d)
            else:
                iia = DAS.refit_fixed(bb, pairs, "block2_attn", r, 701, epochs=scale.e(100)).test_iia
            res.setdefault(r, []).append(iia)
    save("three_input", res)
    return res


# ----------------------------------------------------------------------------- D.3, D.4, D.9
def distance_and_alternatives(arch: str = "transformer", scale: Scale = FULL) -> Dict:
    """IIA by attractor distance (Table 15) and the within-phenomenon alternative SCMs
    (Table 16; paired t over five seeds after averaging three DAS runs per seed)."""
    res: Dict = {"distance": {}, "alternatives": {}}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config(arch, seed, scale))
            bb = ScratchBackbone(model, vocab)
            for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
                pairs = das_pairs(ph, data, var, scale); site = site_for(model, kind)
                runs = [DAS.search(bb, pairs, s, sites=[site], epochs=scale.e(100)) for s in FROM_SCRATCH_DAS_SEEDS]
                if ph == "agreement":
                    res["distance"].setdefault(var, {})[seed] = DAS.iia_by_stratum(bb, pairs["test"], runs[0])
                if kind == "atomic":
                    res["alternatives"].setdefault(f"{ph}/prescribed", []).append(np.mean([r.test_iia for r in runs]))
                    for alt in P.ALTERNATIVES[ph]:
                        scm = compile_program(P.alternative(ph, alt))
                        rel = {k: [PR.Pair(p.base, p.source, var, PR.label_pair(scm, p.base, p.source, var)[0],
                                           p.base_output, p.family, p.load) for p in v] for k, v in pairs.items()}
                        vals = [DAS.refit_fixed(bb, rel, site, r.rank, r.seed, epochs=scale.e(100)).test_iia for r in runs]
                        res["alternatives"].setdefault(f"{ph}/{alt}", []).append(float(np.mean(vals)))
    for key, vals in list(res["alternatives"].items()):
        ph = key.split("/")[0]
        if not key.endswith("prescribed"):
            res["alternatives"][key + ":paired_t"] = ST.paired_t(res["alternatives"][f"{ph}/prescribed"], vals)
    save(f"distance_alternatives_{arch}", res)
    return res


def transfer(scale: Scale = FULL) -> Dict:
    """Appendix D.9: the rotation, rank and layer fitted on a source phenomenon's checkpoint,
    applied unchanged to the same-seed checkpoint of a target phenomenon at the target's
    consumption position, with the target's SCM and data."""
    res: Dict = {}
    fitted, models, pools = {}, {}, {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config("transformer", seed, scale))
            models[(ph, seed)] = ScratchBackbone(model, vocab)
            for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
                pools[(ph, kind)] = pools.get((ph, kind)) or das_pairs(ph, data, var, scale)
                fitted[(ph, seed, kind)] = DAS.refit_fixed(models[(ph, seed)], pools[(ph, kind)],
                                                           site_for(model, kind), 1 if kind == "atomic" else 2, 701,
                                                           epochs=scale.e(100))
    for src in PHENOMENA:
        for tgt in PHENOMENA:
            for kind in ("atomic", "compositional"):
                vals = []
                for seed in MODEL_SEEDS:
                    f = fitted[(src, seed, kind)]
                    vals.append(DAS.evaluate(models[(tgt, seed)], pools[(tgt, kind)]["test"], f.site, f.R, f.rank))
                res.setdefault(f"{src}->{tgt}", {})[kind] = vals
    save("transfer", res)
    return res


# ----------------------------------------------------------------------------- E.1, E.5
def constrained_vs_unconstrained(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    data = core_data("agreement", scale); vocab = vocab_for("agreement")
    for seed in MODEL_SEEDS:
        cfg = config("transformer", seed, scale)
        model, out = get_model("core/agreement", data["train"], vocab, cfg)
        for net, bb in (("trained", ScratchBackbone(model, vocab)),
                        ("random", ScratchBackbone(checkpoint(out, cfg, vocab, 0), vocab))):
            for var, site in (("subj_num", "block1"), ("verb_form", "block2_attn")):
                pairs = das_pairs("agreement", data, var, scale)
                for method in ("das", "mlp", "linear"):
                    vals = []
                    for s in FROM_SCRATCH_DAS_SEEDS:
                        if method == "das":
                            vals.append(DAS.refit_fixed(bb, pairs, site, 2, s, epochs=scale.e(100)).test_iia)
                        else:
                            vals.append(UNC.fit_map(bb, pairs, site, method, s, epochs=scale.e(100))[1])
                    res.setdefault(f"{net}/{var}/{method}", []).append(float(np.mean(vals)))
    save("constrained_vs_unconstrained", res)
    return res


def probing(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph)
        scm = PR.scm_for(ph)
        for seed in MODEL_SEEDS:
            cfg = config("transformer", seed, scale)
            model, out = get_model(f"core/{ph}", data["train"], vocab, cfg)
            for net, m in (("trained", model), ("random", checkpoint(out, cfg, vocab, 0))):
                bb = ScratchBackbone(m, vocab)
                for kind, var in zip(("atomic", "compositional"), PR.VARIABLES[ph]):
                    site = site_for(m, kind)
                    lab = lambda ss: [scm.value(PR._env(s), var, s.target) for s in ss]
                    tr, te = data["train"][:scale.n(20_000)], data["test"]
                    clf, st = PROBES.fit_probe(bb.vectors(tr, site).cpu().numpy(), lab(tr))
                    acc = PROBES.probe_accuracy(clf, st, bb.vectors(te, site).cpu().numpy(), lab(te))
                    res.setdefault(f"{net}/{ph}/{var}", []).append(acc)
    save("probing", res)
    return res


# ----------------------------------------------------------------------------- F.1 joint alignment
def joint(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph); scm = PR.scm_for(ph)
        a_var, c_var = PR.VARIABLES[ph]
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config("transformer", seed, scale))
            bb = ScratchBackbone(model, vocab)
            A = das_pairs(ph, data, a_var, scale)
            C = das_pairs(ph, data, c_var, scale)
            rng = random.Random(seed)
            Tr = {k: JOINT.make_triples(A[k], C[k], scm, rng) for k in A}
            J = JOINT.fit_joint(bb, A, C, Tr, "block1", site_for(model, "compositional"), 701, epochs=scale.e(100))
            train_s = data["train"][:scale.n(20_000)]
            dec = JOINT.fit_mediation_decoder(bb, J, train_s, [scm.value(PR._env(s), c_var, s.target) for s in train_s])
            r = JOINT.evaluate_joint(bb, J, A["test"], C["test"], Tr["test"], dec)
            for k, v in r.items():
                res.setdefault(k, []).append(v)
    save("joint_scratch", res)
    return res


# ----------------------------------------------------------------------------- C.5, D.5, D.10, F.5: SAEs
def sae_suite(scale: Scale = FULL, archs=SAE.ARCHS, expansions=(8,), ks=(1, 2, 4, 8, 16, 32)) -> Dict:
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale); vocab = vocab_for(ph); scm = PR.scm_for(ph)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"core/{ph}", data["train"], vocab, config("transformer", seed, scale))
            bb = ScratchBackbone(model, vocab)
            for layer, (kind, var) in enumerate(zip(("atomic", "compositional"), PR.VARIABLES[ph]), 1):
                site = site_for(model, kind)
                H = bb.vectors(data["train"], site); Hd = bb.vectors(data["val"], site)
                u = np.array([scm.value(PR._env(s), var, s.target) for s in data["train"]], float)
                pairs = das_pairs(ph, data, var, scale)
                for k in ks:
                    units = SX.select_units(H.cpu().numpy(), u, k)
                    res.setdefault(f"neuron/{kind}/{k}", []).append(
                        SX.interchange_iia(bb, pairs["test"], site, lambda a, b, un=units: SX.neuron_patch(a, b, un)))
                for arch in archs:
                    for E in expansions:
                        sae, norm = SAE.train_sae(H, arch, E, SAE.sae_seed(seed, ph, layer, arch, E),
                                                  steps=scale.s(50_000))
                        Z = sae.encode(norm(H)).detach().cpu().numpy()
                        res.setdefault(f"quality/{arch}/{E}", []).append(SAE.quality(sae, norm, Hd))
                        for k in ks:
                            units = SX.select_units(Z, u, k)
                            res.setdefault(f"{arch}/{E}/{kind}/{k}", []).append(SX.interchange_iia(
                                bb, pairs["test"], site, lambda a, b, s_=sae, n_=norm, un=units: SX.feature_patch(s_, n_, a, b, un)))
                        if E == 8:
                            res.setdefault(f"maxcorr/{arch}/{kind}", []).append(SX.max_correlation_test(Z, u, n_perm=scale.n(1_000)))
                            if kind == "compositional":
                                units32 = SX.select_units(Z, u, 32)
                                res.setdefault(f"span/{arch}", []).append(
                                    SX.span_das(bb, pairs, site, sae, norm, units32, epochs=scale.e(100))[1])
    save("sae_suite", res)
    return res


def boolean_coverage(scale: Scale = FULL) -> Dict:
    """Appendix D.10 / F.5: seven two-input Boolean functions, each learned by its own 2L
    Transformer (shared hyperparameters, context 8); 512 test pairs (256 positive) per seed."""
    res: Dict = {}
    vocab = vocab_for("boolean")
    for name, (_, f) in P.BOOLEAN_FUNCTIONS.items():
        fn = lambda a, b, f=f: int(bool(f(np.bool_(a), np.bool_(b))))
        data = D.boolean_dataset(fn, tuple(scale.n(x) for x in (32_000, 4_000, 8_000)))
        scm = PR.scm_for("boolean", boolean_fn=name)
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"boolean/{name}", data["train"], vocab, config("transformer", seed, scale, max_len=8))
            bb = ScratchBackbone(model, vocab)
            rng = random.Random(seed)
            pairs = PR.split_by_family(PR.make_pairs(data["test"], "out", rng, scm), rng)
            pairs["test"] = pairs["test"][:512]
            site = "block2_attn"
            res.setdefault(f"{name}/das_r2", []).append(DAS.refit_fixed(bb, pairs, site, 2, 701, epochs=scale.e(100)).test_iia)
            H = bb.vectors(data["train"], site)
            u = np.array([s.label for s in data["train"]], float)
            for arch in SAE.ARCHS:
                sae, norm = SAE.train_sae(H, arch, 8, seed, steps=scale.s(50_000))
                Z = sae.encode(norm(H)).detach().cpu().numpy()
                for k in (1, 8):
                    un = SX.select_units(Z, u, k)
                    res.setdefault(f"{name}/{arch}/{k}", []).append(SX.interchange_iia(
                        bb, pairs["test"], site, lambda a, b: SX.feature_patch(sae, norm, a, b, un)))
            un = SX.select_units(H.cpu().numpy(), u, 1)
            res.setdefault(f"{name}/neuron/1", []).append(
                SX.interchange_iia(bb, pairs["test"], site, lambda a, b: SX.neuron_patch(a, b, un)))
    save("boolean_coverage", res)
    return res


# ----------------------------------------------------------------------------- Figure 3, E.7, F.3, Figure 5
def trajectories(scale: Scale = FULL, regime: str = "label_only") -> Dict:
    """Per-variable IIA every 500 steps (fixed final site and rank; R refit from DAS seeds
    701/809/907 at every checkpoint), crossing steps, lags and grokking points.

    regime: label_only (core data) | aux (label + 0.3 x atomic loss) | label_input (label token),
    the last two on the held-out noun-verb combinations of Appendix F.3."""
    res: Dict = {}
    for ph in PHENOMENA:
        data = core_data(ph, scale) if regime == "label_only" else D.combination_dataset(ph)
        vocab = vocab_for(ph, label_tokens=regime == "label_input")
        scm = PR.scm_for(ph); a_var, c_var = PR.VARIABLES[ph]
        for seed in MODEL_SEEDS:
            cfg = config("transformer", seed, scale, aux_weight=0.3 if regime == "aux" else 0.0,
                         label_as_input=regime == "label_input", max_len=31 if regime == "label_input" else 30)
            atomic = lambda s: scm.value(PR._env(s), a_var, s.target)
            model, out = get_model(f"traj/{regime}/{ph}", data["train"], vocab, cfg,
                                   eval_sets={"train": data["train"][:5000], "test": data["test"]},
                                   atomic_labels=atomic if regime == "aux" else None)
            pools = {kind: das_pairs(ph, data, var, scale) for kind, var in (("atomic", a_var), ("compositional", c_var))}
            final = {kind: DAS.search(ScratchBackbone(model, vocab, label_as_input=cfg.label_as_input),
                                      pools[kind], 701, sites=[site_for(model, kind)], epochs=scale.e(100))
                     for kind in pools}
            steps = [r["step"] for r in model.training_log]
            curves = {"atomic": [], "compositional": []}
            for step in steps:
                bb = ScratchBackbone(checkpoint(out, cfg, vocab, step), vocab, label_as_input=cfg.label_as_input)
                for kind in ("atomic", "compositional"):
                    pairs = pools[kind]
                    vals = [DAS.refit_fixed(bb, pairs, final[kind].site, final[kind].rank, s, epochs=scale.e(100)).test_iia
                            for s in FROM_SCRATCH_DAS_SEEDS]
                    curves[kind].append(float(np.mean(vals)))
            a90 = CROSS.crossing_step(steps, curves["atomic"]); c90 = CROSS.crossing_step(steps, curves["compositional"])
            acc = [r.get("test_accuracy") for r in model.training_log]
            tr = [r.get("train_accuracy") for r in model.training_log]
            res[f"{ph}/{seed}"] = {"steps": steps, "curves": curves, "test_accuracy": acc, "train_accuracy": tr,
                                   "atomic_cross": a90, "compositional_cross": c90, "lag": CROSS.lag(a90, c90),
                                   "grokking_95": CROSS.crossing_step(steps, acc, 0.95),
                                   "grokking_99": CROSS.crossing_step(steps, acc, 0.99),
                                   "train_fit_99": CROSS.crossing_step(steps, tr, 0.99)}
    lags = [v["lag"] for v in res.values() if v["lag"] is not None]
    res["summary"] = {"sign_test": ST.exact_sign_test(lags), "wilcoxon": ST.wilcoxon(lags) if lags else None,
                      "median_lag": float(np.median(lags)) if lags else None}
    save(f"trajectories_{regime}", res)
    return res


def excluded_loss_curve(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    data = core_data("agreement", scale); vocab = vocab_for("agreement")
    scm = PR.scm_for("agreement")
    for seed in MODEL_SEEDS:
        cfg = config("transformer", seed, scale)
        model, out = get_model("core/agreement", data["train"], vocab, cfg)
        tr = data["train"][:scale.n(10_000)]; te = data["test"][:scale.n(10_000)]
        subj = [s.meta["subj_pl"] for s in tr]
        verb = [int(PR._env(s)["Q_v_pl"][s.target]) for s in tr]
        for rec in model.training_log:
            m = checkpoint(out, cfg, vocab, rec["step"])
            res.setdefault(str(seed), []).append((rec["step"], CROSS.excluded_loss(m, vocab, tr, te, subj, verb, cfg.device)))
    save("excluded_loss", res)
    return res


# ----------------------------------------------------------------------------- D.8 separation suite
def concord_stress(scale: Scale = FULL) -> Dict:
    res: Dict = {}
    data = D.concord_stress_dataset(tuple(scale.n(x) for x in (45_000, 4_500, 9_000)))
    vocab = vocab_for("concord_stress"); scm = PR.scm_for("concord_stress")
    by_level = {}
    for s in data["test"]:
        by_level.setdefault(s.load, []).append(s)
    for arch in ARCHS:
        for seed in MODEL_SEEDS:
            model, _ = get_model(f"concord_stress", data["train"], vocab, config(arch, seed, scale))
            bb = ScratchBackbone(model, vocab)
            from ..models.training import accuracy
            for lvl, ss in sorted(by_level.items()):
                res.setdefault(f"{arch}/accuracy/{lvl}", []).append(accuracy(model, ss, vocab, bb.device))
                rng = random.Random(seed * 10 + lvl)
                pairs = PR.split_by_family(PR.make_pairs(ss, "conc_ok", rng, scm), rng)
                res.setdefault(f"{arch}/iia/{lvl}", []).append(
                    DAS.refit_fixed(bb, pairs, site_for(model, "compositional"), 2, 701, epochs=scale.e(100)).test_iia)
    for arch in ARCHS:
        acc0, acc8 = np.array(res[f"{arch}/accuracy/0"]), np.array(res[f"{arch}/accuracy/8"])
        res[f"{arch}/accuracy_drop"] = (acc0 - acc8).tolist()
        res[f"{arch}/tost_p"] = ST.tost(acc0 - acc8, 0.04)
        res[f"{arch}/degrade_p"] = ST.one_sided_t_greater(acc0 - acc8)
    save("concord_stress", res)
    return res
