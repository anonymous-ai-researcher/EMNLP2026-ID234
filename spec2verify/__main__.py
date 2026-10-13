"""Command-line entry point: ``python -m spec2verify <command> [options]``.

Each command reproduces one table, figure or appendix analysis of the paper (see
README.md for the mapping).  ``--quick`` shrinks data, steps and DAS epochs for an
end-to-end smoke run; omit it to run the full protocol.
"""

from __future__ import annotations

import argparse

from .experiments import common


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spec2verify")
    ap.add_argument("command", help="see README.md")
    ap.add_argument("--quick", action="store_true", help="small end-to-end smoke run")
    ap.add_argument("--model", default="pythia-1.4b", help="pre-trained model short name")
    ap.add_argument("--arch", default="transformer", choices=("transformer", "lstm", "mamba"))
    ap.add_argument("--regime", default="label_only", choices=("label_only", "aux", "label_input"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--blimp-dir", default=None, help="local BLiMP JSONL files (default: Hugging Face)")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated model seeds (default: all five)")
    a = ap.parse_args(argv)
    scale = common.QUICK if a.quick else common.FULL
    dev = a.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    from .experiments import architecture as AR
    from .experiments import failure as FA
    from .experiments import pretrained as PT
    from .experiments import scratch as SC
    from .experiments import structure as STR
    if a.seeds:
        seeds = tuple(int(x) for x in a.seeds.split(","))
        for mod in (SC, STR, AR):
            mod.MODEL_SEEDS = seeds
    commands = {
        # Table 2 (2-layer models), Table 18 (wrong SCMs), Appendix C.3 variance decomposition
        "table2-scratch": lambda: SC.table2(a.arch, scale),
        # Table 2 (pre-trained), Tables 12, 14, 43; Figure 2 data; rank stability
        "table2-pretrained": lambda: PT.core_suite(a.model, scale, dev),
        "wrong-scm-pretrained": lambda: PT.wrong_scm(a.model, scale, dev),
        # Section 5.2 / Table 3 / Appendix F.2
        "structure-english": lambda: STR.english_scratch(scale),
        "structure-german": lambda: STR.german_scratch(scale),
        "structure-english-pretrained": lambda: STR.english_pretrained(a.model, scale, dev),
        "structure-german-pretrained": lambda: STR.german_pretrained(a.model, scale, dev),
        # Section 5.3 / Figure 3 / Table 29 / Table 35 / Figure 5
        "trajectories": lambda: SC.trajectories(scale, a.regime),
        "excluded-loss": lambda: SC.excluded_loss_curve(scale),
        # Section 5.4 / Table 4 / Appendices D.5, D.10, F.5, F.6
        "sae": lambda: SC.sae_suite(scale, expansions=(4, 8, 16, 32)),
        "boolean": lambda: SC.boolean_coverage(scale),
        "rank-ablation": lambda: SC.rank_ablation(scale),
        "rank-by-layer": lambda: SC.rank_by_layer(scale),
        "three-input": lambda: SC.three_input(scale),
        # Section 5.5 / Figure 4 / Tables 41-42
        "arch-tune": lambda: [AR.tune(x, scale) for x in ("transformer", "lstm", "mamba")],
        "arch-sweep": lambda: AR.sweep(a.arch, scale),
        "arch-growth": lambda: AR.growth(a.arch),
        "mamba-probe": lambda: AR.mamba_state_probe(scale),
        # Appendix D
        "distance-alternatives": lambda: SC.distance_and_alternatives(a.arch, scale),
        "transfer": lambda: SC.transfer(scale),
        "concord-stress": lambda: SC.concord_stress(scale),
        # Appendix E
        "unconstrained": lambda: SC.constrained_vs_unconstrained(scale),
        "probing": lambda: SC.probing(scale),
        "blimp": lambda: PT.blimp_suite(a.model, scale, dev, a.blimp_dir),
        # Appendix F
        "joint": lambda: SC.joint(scale),
        "joint-pretrained": lambda: PT.joint_pretrained(a.model, scale, dev),
        "failure": lambda: FA.run(a.model, scale, dev),
        "attention": lambda: PT.attention_suite(a.model, scale, dev),
        "figures": lambda: __import__("spec2verify.figures", fromlist=["ALL"]).ALL[a.figure]()
        if a.figure else [f() for f in __import__("spec2verify.figures", fromlist=["ALL"]).ALL.values()],
    }
    if a.command not in commands:
        ap.error(f"unknown command {a.command!r}; choose from: " + ", ".join(commands))
    commands[a.command]()


if __name__ == "__main__":
    main()
