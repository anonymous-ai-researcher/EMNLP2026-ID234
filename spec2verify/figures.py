"""Figures 2-6 from the result files written by the experiment runners.

Colors follow the paper's convention: blue for atomic variables (lookups), red
for compositional variables (combinations), gray for inputs, chance, reference
lines, controls and baselines; in Figure 2(a), light for layer 1 and dark for
layer 2.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .experiments.common import RESULTS, load

BLUE, RED, GRAY = "#2f6db3", "#c0392b", "#8c8c8c"
TITLE = {"agreement": "Agreement", "npi": "NPI", "binding": "Binding", "concord": "Concord"}
ARCH_COLORS = {"transformer": "#1b9e77", "lstm": "#d95f02", "mamba": "#7570b3"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _out(name: str) -> str:
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    return os.path.join(RESULTS, "figures", name)


def figure2_perlayer(model: str = "pythia-1.4b"):
    """Per-layer development IIA (best rank per layer), atomic vs compositional variables."""
    plt = _plt(); r = load(f"pretrained_core_{model}")
    fig, axes = plt.subplots(1, 4, figsize=(7, 1.8), sharey=True)
    for ax, ph in zip(axes, ("agreement", "npi", "binding", "concord")):
        for var, color in zip(r_vars(ph), (BLUE, RED)):
            curves = []
            for run in r[f"{ph}/{var}"]["per_layer_dev"]:
                best: Dict[int, float] = {}
                for site, rank, dev in run:
                    l = int(site.replace("layer", "")); best[l] = max(best.get(l, 0.0), dev)
                curves.append([best[l] for l in sorted(best)])
            c = np.array(curves); x = np.arange(1, c.shape[1] + 1)
            ax.plot(x, c.mean(0), color=color, lw=1.2, ls="-" if color == BLUE else "--")
            ax.fill_between(x, c.mean(0) - c.std(0, ddof=1), c.mean(0) + c.std(0, ddof=1), color=color, alpha=0.2, lw=0)
        ax.axhline(0.5, color=GRAY, lw=0.6, ls=":"); ax.set_title(TITLE[ph]); ax.set_xlabel("Layer")
    axes[0].set_ylabel("IIA")
    fig.tight_layout(); fig.savefig(_out("fig_perlayer.pdf")); plt.close(fig)


def r_vars(ph: str):
    from .data.pairs import VARIABLES
    return VARIABLES[ph]


def figure3_trajectory(regime: str = "label_only"):
    plt = _plt(); r = load(f"trajectories_{regime}")
    fig, axes = plt.subplots(1, 4, figsize=(7, 1.8), sharey=True)
    for ax, ph in zip(axes, ("agreement", "npi", "binding", "concord")):
        runs = [v for k, v in r.items() if k.startswith(ph + "/")]
        steps = np.array(runs[0]["steps"])
        for kind, color in (("atomic", BLUE), ("compositional", RED)):
            c = np.array([v["curves"][kind] for v in runs])
            ax.plot(steps, c.mean(0), color=color, lw=1.2, ls="-" if kind == "atomic" else "--")
            ax.fill_between(steps, c.mean(0) - c.std(0, ddof=1), c.mean(0) + c.std(0, ddof=1), color=color, alpha=0.2, lw=0)
        acc = np.array([v["test_accuracy"] for v in runs]).mean(0)
        ax.plot(steps, acc, color=GRAY, lw=0.8)
        g = [v["grokking_95"] for v in runs if v["grokking_95"] is not None]
        if g:
            ax.axvline(500 * round(np.mean(g) / 500), color=GRAY, lw=0.6, ls=":")
        ax.set_title(TITLE[ph]); ax.set_xlabel("Step")
    axes[0].set_ylabel("IIA / accuracy")
    fig.tight_layout(); fig.savefig(_out("fig_iia_trajectory.pdf")); plt.close(fig)


def figure4_distance():
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.2))
    for arch, color in ARCH_COLORS.items():
        try:
            g = load(f"growth_{arch}")["main"]; sw = load(f"arch_sweep_{arch}")
        except FileNotFoundError:
            continue
        ds = [d for d, p in enumerate(g["pstar"]) if p is not None]
        b.semilogy(ds, [g["pstar"][d] for d in ds], "o-", color=color, ms=3, label=arch)
        accs = [np.mean(sw["accuracy"][f"128/{d}"]) for d in range(17)]
        a.plot(range(17), accs, "o-", color=color, ms=3, label=arch)
    a.set_xlabel("Attractor distance"); a.set_ylabel("Accuracy (width 128)"); a.axhline(0.9, color=GRAY, lw=0.6, ls=":")
    b.set_xlabel("Attractor distance"); b.set_ylabel(r"$P^*(d)$"); b.legend(frameon=False)
    fig.tight_layout(); fig.savefig(_out("fig_succinctness.pdf")); plt.close(fig)


def figure5_excluded_loss():
    plt = _plt(); traj = load("trajectories_label_only"); ex = load("excluded_loss")
    runs = [v for k, v in traj.items() if k.startswith("agreement/")]
    steps = np.array(runs[0]["steps"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.0))
    for kind, color in (("atomic", BLUE), ("compositional", RED)):
        c = np.array([v["curves"][kind] for v in runs])
        a.plot(steps, c.mean(0), color=color, ls="-" if kind == "atomic" else "--")
    L = np.array([[x[1] for x in v] for v in ex.values()])
    b.plot([x[0] for x in next(iter(ex.values()))], L.mean(0), color=GRAY)
    g = [v["grokking_95"] for v in runs if v["grokking_95"] is not None]
    for ax in (a, b):
        if g:
            ax.axvline(500 * round(np.mean(g) / 500), color=GRAY, lw=0.6, ls=":")
        ax.set_xlabel("Step")
    a.set_ylabel("IIA"); b.set_ylabel("Excluded loss")
    fig.tight_layout(); fig.savefig(_out("fig_trajectory_comparison.pdf")); plt.close(fig)


def figure6_scaling(models=("pythia-160m", "pythia-410m", "pythia-1.4b")):
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(7, 1.8), sharey=True)
    sizes = [160, 410, 1400]
    for ax, ph in zip(axes, ("agreement", "npi", "binding", "concord")):
        for var, color in zip(r_vars(ph), (BLUE, RED)):
            vals = [load(f"pretrained_core_{m}")[f"{ph}/{var}"]["test_iia"] for m in models]
            ax.errorbar(sizes, [np.mean(v) for v in vals], [np.std(v, ddof=1) for v in vals], color=color,
                        ls="-" if color == BLUE else "--", marker="o", ms=3, capsize=2)
        ax.set_xscale("log"); ax.set_title(TITLE[ph]); ax.set_xlabel("Parameters (M)")
    axes[0].set_ylabel("IIA")
    fig.tight_layout(); fig.savefig(_out("fig_scaling.pdf")); plt.close(fig)


ALL = {"2": figure2_perlayer, "3": figure3_trajectory, "4": figure4_distance, "5": figure5_excluded_loss,
       "6": figure6_scaling}
