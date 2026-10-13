# spec2verify

Code for **"Look Up, Then Combine: Formal Specifications as Causal Hypotheses for
Syntactic Dependencies in Transformers"**.

A syntactic rule is written as a Boolean RASP (B-RASP) program, compiled into a
structural causal model (SCM), and tested against trained networks with
interchange interventions found by Distributed Alignment Search (DAS).  Each
program has an *atomic* variable (one attention operation, a lookup) and a
*compositional* variable (a pointwise Boolean function, a combination).  The
repository covers subject–verb agreement, NPI licensing, reflexive binding and
determiner–noun concord, plus German verb-final clauses, the decisive English
constructions and the controlled Boolean tasks.

No data files are included: every dataset is generated deterministically from
the templates and seeds in `spec2verify/data`, and BLiMP and the pre-trained
checkpoints are downloaded from their public sources.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10
pip install -r requirements.txt
pip install -e .
# optional, CUDA only: the fused Mamba-1 kernel (otherwise a plain PyTorch scan is used)
pip install mamba-ssm==1.2.2 causal-conv1d==1.2.0.post2 && export SPEC2VERIFY_MAMBA_SSM=1
python -m pytest -q tests                                 # 23 tests, about one minute on CPU
```

The experiments were run with Python 3.10.13, PyTorch 2.1.2 (CUDA 12.1),
`transformers` 4.44.2, `tokenizers` 0.19.1, NumPy 1.26.4, SciPy 1.11.4 and
scikit-learn 1.3.2 on NVIDIA A100 GPUs (40 GB), with TF32 and cuDNN autotuning
disabled.  The compute budget is about 6,480 A100 GPU hours in total.

## Repository layout

```
spec2verify/
  brasp/        B-RASP semantics (core.py), the programs of Section 3 and Appendix A
                (programs.py), and their compilation into SCMs with interchange
                interventions (scm.py)
  data/         lexica (Table 5), vocabularies (135/137/139/167/40/66 tokens), CFG
                templates, the 40 natural-text templates per phenomenon for the
                pre-trained models, distribution shifts, dataset assembly with
                sentence-family splits, and counterfactual pairs (Appendix B)
  models/       2-layer Transformer, LSTM and Mamba-1 classifiers whose parameter
                counts equal the formulas of Appendix F.7; training (Table 8)
  alignment/    DAS with the Cayley parameterization, rank/layer search, negative
                controls; joint alignment (F.1); unconstrained maps (E.1); probes (E.5)
  sae/          TopK, JumpReLU and Gated SAEs; interchange scoring, permutation null,
                DAS in the span of selected decoder directions (C.5, D.5, D.10, F.5)
  pretrained/   Hugging Face models (C.2, C.4), BLiMP (B.2, E.3), head ablation and
                attention mass (F.8)
  analysis/     statistical tests (C.7), parameter-growth fits (F.7), failure
                prediction (F.4), crossing steps and excluded loss (E.2, E.7)
  experiments/  one runner per table, figure or appendix analysis
  figures.py    Figures 2-6 from the result files
tests/          unit tests (semantics, data, models, alignment, pre-trained interface)
```

## Reproducing the paper

All commands write JSON results to `results/` (set `SPEC2VERIFY_RESULTS`) and
cache checkpoints in `runs/` (set `SPEC2VERIFY_RUNS`).  `--quick` runs a small
end-to-end version of any command (fewer sentences, steps and DAS epochs);
`--seeds 1103` restricts the five model seeds.

| Paper | Command |
|---|---|
| Table 2, 2-layer models; Table 18 (wrong SCMs); Appendix C.3 variance decomposition | `python -m spec2verify table2-scratch --arch {transformer,lstm,mamba}` |
| Table 2, pre-trained models; Tables 12, 14 and 43; Figure 2 data; rank stability (D.1) | `python -m spec2verify table2-pretrained --model pythia-1.4b` (and the other four English models) |
| Wrong-SCM control for the pre-trained models | `python -m spec2verify wrong-scm-pretrained --model ...` |
| Section 5.2, Table 3, Tables 32–34 (2-layer models) | `structure-english`, `structure-german` |
| Section 5.2, pre-trained models | `structure-english-pretrained --model ...`, `structure-german-pretrained --model ...` |
| Figure 3, Table 29 (crossing steps), Table 35 (training objectives) | `trajectories --regime {label_only,aux,label_input}` |
| Figure 5 (excluded loss) | `excluded-loss` (after `trajectories`) |
| Table 4, Tables 17, 22–24, 37–39 (SAEs) | `sae`, `boolean` |
| Table 13 (rank ablation), Table 40 (rank by layer), three-input NPI (D.1) | `rank-ablation`, `rank-by-layer`, `three-input` |
| Figure 4, Tables 41–42 | `arch-tune`, `arch-sweep --arch ...`, `arch-growth --arch ...`, `mamba-probe` |
| Table 15 (IIA by distance), Table 16 (alternative SCMs) | `distance-alternatives` |
| Table 21 (transfer) | `transfer` |
| Table 20 (separation suite) | `concord-stress` |
| Table 25 (constrained vs. unconstrained) | `unconstrained` |
| Table 27 (probing) | `probing` |
| Tables 6, 26, 30 (BLiMP) | `blimp --model ...` |
| Table 31 (joint alignment) | `joint`, `joint-pretrained --model ...` |
| Table 36 (failure prediction) | `failure --model pythia-1.4b` (after `table2-pretrained`) |
| Attention analysis (F.8) | `attention --model ...` |
| Figures 2–6 | `figures` (or `figures --figure 3`) |

Model short names: `pythia-160m`, `pythia-410m`, `pythia-1.4b`, `pythia-2.8b`,
`pythia-6.9b`, `llama-3.2-1b`, `llama-3.2-3b`, `llama-3.1-8b`,
`llama-3.1-8b-instruct`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`,
`gemma-2-2b` (Table 9; Pythia at revision `step143000`).  The Llama and Gemma
checkpoints require accepting their licenses on Hugging Face.

## Protocol summary

**Data (Appendix B).**  Agreement 50,000/5,000/10,000 sentences
(train/validation/test); NPI, binding and concord 40,000/4,000/8,000 each;
balanced labels; agreement and binding spread evenly over structural loads 0–8.
Generation seeds: 42 (agreement, binding, concord), 20270723 (scope-sensitive
NPI), 20270722 (separation suite).  Splits are made by sentence family, so no
sentence or minimal pair occurs in two splits; when a stratum has fewer distinct
families than its quota, records are drawn with replacement inside the split.
DAS pairs come from fresh families (≥ 5,000 pairs per load level, split 60/20/20
by family: 3,000/1,000/1,000), with balanced counterfactual labels computed by the
SCM.  Half of the pairs edit the critical token (the variable changes); the other
half use a neutral edit that keeps the variable and the output, so an intervention
without effect scores 0.5 (`experiments/scratch.py: UNCHANGED_FRACTION`).  Pre-trained models use 6,000/2,000/2,000 pairs per variable from 40
templates and keep only bases and sources the model handles correctly.

**Models (Appendix C.1).**  Two layers, width 128; causal pre-LN Transformer
(4 heads, FFN 512, GELU, learned positions, Xavier-uniform weights, N(0, 0.02²)
embeddings); unidirectional LSTM; Mamba-1 (state 16, expansion 2, convolution 4,
dt rank 8, pre-RMSNorm).  AdamW (lr 1e-3, weight decay 1.0 on all parameters),
500 warm-up steps and cosine decay, batch 64, 50,000 steps, gradient clip 1.0,
no dropout, checkpoints every 500 steps; model seeds 1103, 2207, 3319, 4421, 5531.

**DAS (Appendix C.3).**  R = R0 (I − Ω)(I + Ω)⁻¹, Ω = A − Aᵀ; Adam lr 5e-3, 100
epochs, batch 256, best development epoch; ranks {1, 2, 4, 8}; all block outputs
of the pre-trained models (4L configurations per variable), fixed sites in the
2-layer models (first-block output for atomic variables, the residual stream
entering the last MLP for compositional ones); interventions at the position
where the variable is consumed.  DAS seeds 701, 809, 907, 1013, 1109 (the first
three for the from-scratch models).  Controls: wrong SCM, random initialization
(step 0, or the pre-trained architecture rebuilt from its configuration with
seeds 1103–5531), shuffled counterfactual labels; competitors and controls reuse
the selected site and rank and refit only R.

**Pre-trained read-out.**  The language model is read where it predicts the
critical word (verb, NPI, reflexive or noun): the intervention position is the
last token before that word (the last subword of a named word when an analysis
asks for it, such as the German noun for `can_nom`), and the network's discrete
output compares the total log-probability of the word's two candidate forms;
critical words with more than three subwords are filtered.  Behavioral accuracy
compares sentence log-probabilities (one BOS for Llama and Gemma, none for Pythia
and Qwen, no length normalization, ties broken by pair-ID parity).

**SAEs (Appendix C.5).**  One SAE per seed, phenomenon, site, architecture and
expansion; 50,000 updates of batch 4,096; TopK K = 32; JumpReLU and Gated with
sparsity coefficients 1e-4 and 3e-4 at 8× (chosen to match mean L0 = 32);
interchange scores keep the base reconstruction error.

## License

MIT.
