<div align="center">

# Look Up, Then Combine

**Formal Specifications as Causal Hypotheses for Syntactic Dependencies in Transformers**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers 4.44](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.44-ffcc4d.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[The idea](#-the-idea-in-one-example) ·
[Findings](#-what-we-find) ·
[Quick start](#-quick-start) ·
[Code map](#-how-the-code-follows-the-pipeline) ·
[Reproduce](#-reproducing-the-paper) ·
[Details](#-protocol-details)

</div>

---

Interpretability usually starts from a hand-written guess about what a network computes.
This repository starts from a **formal program** instead. We write each syntactic rule
in **B-RASP**, a programming language whose operations compile into Transformer layers,
turn the program into a **structural causal model (SCM)**, and then ask whether a trained
network contains internal states that behave like the program's variables under
**interchange interventions** found by **Distributed Alignment Search (DAS)**.

Every B-RASP program is built from just two operations:

| Operation | What it does | Variable it produces | Example |
|---|---|---|---|
| 🔍 **Look up** (attention) | finds one token and reads a feature from it, at any distance | **atomic** | the number of the subject |
| 🧩 **Combine** (pointwise) | applies a Boolean function to values at the same position | **compositional** | whether subject and verb numbers match |

The rest of this page walks through one example, summarizes what the experiments show,
and then explains how to run and navigate the code.

## 💡 The idea in one example

Take subject–verb agreement in

> *The **birds** near the **lake** **sing***

The verb must match *birds*, not the closer noun *lake* (an *attractor*).

### Step 1. Write the rule as a program

```text
subj_num(i)  := ◁_j [ j < i , Q_noun(j) ]  Q_n_pl(j) : 0     look up the leftmost noun before i, read its number
verb_form(i) := subj_num(i) ↔ Q_v_pl(i)                     combine: does the verb's number match?
```

At the verb *sing*, `subj_num = 1` (plural; *lake* is skipped) and `verb_form = 1` (grammatical).

### Step 2. Compile it into a causal model

Each operation becomes one endogenous variable whose parents are the vectors it reads:

```text
 Q_noun, Q_n_pl ──► subj_num ──► verb_form ──► grammatical?
                    (atomic)   (compositional)
         Q_v_pl ─────────────────────┘
```

### Step 3. Ask a counterfactual question

An interchange intervention runs the model on a **base** sentence but replaces one
variable with its value from a **source** sentence. The SCM tells us what *should* happen:

| | Sentence | `subj_num` | `verb_form` |
|---|---|:---:|:---:|
| base | *The birds near the lake sing* | 1 | 1 ✅ |
| source | *The bird near the lake sing* | 0 | 0 ❌ |
| base with `subj_num ← source` | *The birds near the lake sing*, with the subject's number swapped in | **0** | **0** ❌ |

So if a network really stores the subject's number in some subspace, patching that
subspace from the source should make it call the unchanged base sentence
**ungrammatical**, even though the word *birds* is still there.

You can check the SCM side directly:

```python
from lookup_combine.brasp import programs
from lookup_combine.brasp.scm import compile_program
from lookup_combine.data.vocab import english_env

scm = compile_program(programs.agreement())
print(scm.parents)   # {'subj_num': ['Q_noun', 'Q_n_pl'], 'verb_form': ['subj_num', 'Q_v_pl']}

base   = english_env("the birds near the lake sing".split(), "agreement")
source = english_env("the bird near the lake sing".split(), "agreement")
verb = 5                                                   # position of "sing"

scm.value(base, "subj_num", verb)                          # True : plural subject, "lake" skipped
scm.output_at(base, verb)                                  # True : grammatical
scm.interchange(base, {"subj_num": source}, verb)          # False: the counterfactual label
```

### Step 4. Search the network for the variable

DAS learns a rotation `R` of a hidden state and swaps only its first `r` coordinates
between the two runs:

```text
h_int = h_base + Rᵀ Π_r R (h_source − h_base)          R = R₀ (I − Ω)(I + Ω)⁻¹ ,  Ω = A − Aᵀ
```

**Interchange intervention accuracy (IIA)** is the fraction of pairs on which the
patched network gives the label the SCM predicts. Labels are balanced, so chance is
**0.5**. A wrong program, an untrained network or shuffled labels should all stay there.

### Why the two kinds of variable matter

A lookup copies one input feature, so a direction for it exists as soon as the token is
read. A combination like `verb_form` is a biconditional: writing its inputs as
`x, y ∈ {±1}`, it equals `x·y`, which has **no linear term**. For independent uniform
inputs, no direction in their span correlates with it until the network computes it.
This difference shapes when a variable is learned and whether sparse autoencoders find it.

## 📊 What we find

Four English dependencies, each with one lookup and one combination:

| Phenomenon | Grammatical | Ungrammatical | Atomic (look up) | Compositional (combine) |
|---|---|---|---|---|
| Agreement | *The birds near the lake **sing*** | *The birds near the lake **sings*** | `subj_num` | `verb_form` (↔) |
| NPI licensing | ***No** pilots who help the researcher have **ever** responded* | *The pilots who help **no** researcher have **ever** responded* | `has_lic` | `npi_ok` (∨ ¬) |
| Binding | *The boy admired **himself*** | *The boy admired **themselves*** | `ante_num` | `refl_ok` (↔) |
| Concord | ***This** book is good* | ***These** book is good* | `det_num` | `conc_ok` (↔) |

| Question | Answer |
|---|---|
| **Do trained Transformers realize the programs?** | Approximately, yes. One joint alignment recovers both steps (IIA 0.86–0.96), while wrong programs, untrained networks and shuffled labels stay at 0.5. |
| **Do lookups follow structure or position?** | Structure. In *Near the lakes the **bird** by the trees sings*, pre-trained models align with the hierarchical subject (0.85–0.91), not with the leftmost or the nearest noun. In German *dass den Mann die Frauen sehen*, where only case marks the subject, the case-first program wins (0.83–0.91). |
| **Which is learned first?** | Lookups. Trained from scratch, the atomic variable crosses IIA 0.90 first in all 20 phenomenon-seed runs, a median of 5,000 steps earlier. |
| **Can sparse autoencoders find them?** | Only the lookups. A single SAE feature gives 0.89 on atomic but 0.55 on compositional variables, against 0.97 and 0.94 for DAS, with TopK, JumpReLU and Gated SAEs alike. |
| **Does the architecture matter?** | Yes. Reaching 0.90 at attractor distance 8 takes the smallest Transformer we train (32K parameters), but 4.3M for the LSTM and 16M for Mamba, whose requirement grows about 1.8× per intervening phrase. |

## 🚀 Quick start

### 1. Install

```bash
# download and unzip the anonymous repository, then:
cd lookup-combine
python -m venv .venv && source .venv/bin/activate     # Python 3.10
pip install -r requirements.txt
pip install -e .
```

<details>
<summary>Optional: fused Mamba-1 kernel (CUDA only)</summary>

Without it, Mamba runs on a plain PyTorch scan with the same parameters.

```bash
pip install mamba-ssm==1.2.2 causal-conv1d==1.2.0.post2
export LOOKUP_COMBINE_MAMBA_SSM=1
```
</details>

### 2. Check the installation

```bash
python -m pytest -q tests       # 23 tests: B-RASP semantics, data, models, DAS, SAEs, HF interface
```

### 3. Run one experiment end to end

```bash
python -m lookup_combine table2-scratch --arch transformer --quick --seeds 1103 --device cpu
```

`--quick` shrinks the data, training steps and DAS epochs so that the whole pipeline
(data → model → DAS → controls) finishes in a few minutes on a CPU. It checks that
everything runs; its numbers are not meant to match the paper. Drop `--quick` for the
full protocol.

Results are written as JSON to `results/` and checkpoints to `runs/`
(override with `LOOKUP_COMBINE_RESULTS` and `LOOKUP_COMBINE_RUNS`).

## 🧭 How the code follows the pipeline

Each stage of the method has its own subpackage, so reading them in this order follows the paper:

```text
 ① specify         ② compile        ③ generate data      ④ train / load      ⑤ align         ⑥ analyze
 brasp/programs ─► brasp/scm ─────► data/ ─────────────► models/ ──────────► alignment/ ───► analysis/
                                    (templates, pairs)    pretrained/         sae/            figures.py
```

```text
lookup_combine/
├── brasp/        ① B-RASP semantics (core.py) and every program in the paper (programs.py)
│                 ② compilation into SCMs and counterfactual labels (scm.py)
├── data/         ③ lexica, vocabularies, CFG templates, the 40 natural-text templates per
│                    phenomenon, distribution shifts, family-level splits, counterfactual pairs
├── models/       ④ 2-layer Transformer, LSTM and Mamba-1 classifiers; training and tuned settings
├── pretrained/   ④ Hugging Face models, BLiMP, head ablation and attention mass
├── alignment/    ⑤ DAS (Cayley rotation, rank and layer search, controls), joint alignment,
│                    unconstrained maps, linear probes
├── sae/          ⑤ TopK, JumpReLU and Gated SAEs; feature interchange and permutation null
├── analysis/     ⑥ statistical tests, parameter-growth fits, failure prediction, crossing steps
├── experiments/  one runner per table, figure or appendix analysis
├── figures.py    Figures 2–6 from the result files
└── __main__.py   command-line interface
tests/            unit tests
scripts/          reproduce_all.sh
```

**No data files are shipped.** Every dataset is generated deterministically from the
templates and fixed seeds in `data/`; BLiMP and the pre-trained checkpoints are
downloaded from their public sources.

## 🔁 Reproducing the paper

All experiments run through one entry point:

```bash
python -m lookup_combine <command> [--arch ...] [--model ...] [--regime ...] [--seeds ...] [--quick]
```

`scripts/reproduce_all.sh` runs every command in dependency order.

### Main text

| Section | What it shows | Command |
|---|---|---|
| §5.1, Table 2 (from scratch) | IIA per variable and negative controls | `table2-scratch --arch {transformer,lstm,mamba}` |
| §5.1, Table 2 (pre-trained), Figure 2 | IIA per variable, per-layer peaks | `table2-pretrained --model pythia-1.4b` (and the other models) |
| §5.1, joint alignment | one alignment for the whole chain | `joint`, `joint-pretrained --model ...` |
| §5.1, failure prediction | alignment predicts errors | `failure --model pythia-1.4b` (after `table2-pretrained`) |
| §5.2, Table 3 | structural vs. linear lookups | `structure-english`, `structure-german`, `structure-english-pretrained --model ...`, `structure-german-pretrained --model ...` |
| §5.3, Figure 3 | lookups emerge before combinations | `trajectories --regime {label_only,aux,label_input}` |
| §5.4, Table 4 | SAEs find lookups, not combinations | `sae`, `boolean` |
| §5.5, Figure 4 | lookups across distance | `arch-tune`, `arch-sweep --arch ...`, `arch-growth --arch ...`, `mamba-probe` |
| Figures 2–6 | all figures | `figures` (or `figures --figure 3`) |

<details>
<summary><b>Appendix analyses</b></summary>

| Appendix item | Command |
|---|---|
| Table 18 (wrong SCMs), variance decomposition (C.3) | `table2-scratch --arch ...` |
| Tables 12, 14 and 43; rank stability (D.1) | `table2-pretrained --model ...` |
| Wrong-SCM control, pre-trained models | `wrong-scm-pretrained --model ...` |
| Tables 32–34 (decisive constructions) | `structure-english`, `structure-german` |
| Table 29 (crossing steps), Table 35 (training objectives) | `trajectories --regime ...` |
| Figure 5 (excluded loss) | `excluded-loss` (after `trajectories`) |
| Tables 17, 22–24, 37–39 (SAEs and Boolean tasks) | `sae`, `boolean` |
| Table 13 (rank ablation), Table 40 (rank by layer), three-input NPI (D.1) | `rank-ablation`, `rank-by-layer`, `three-input` |
| Tables 41–42 (parameter growth) | `arch-sweep`, `arch-growth`, `mamba-probe` |
| Table 15 (IIA by distance), Table 16 (alternative SCMs) | `distance-alternatives` |
| Table 20 (separation suite) | `concord-stress` |
| Table 21 (cross-phenomenon transfer) | `transfer` |
| Table 25 (constrained vs. unconstrained) | `unconstrained` |
| Table 27 (probing vs. DAS) | `probing` |
| Tables 6, 26, 30 (BLiMP) | `blimp --model ...` |
| Table 31 (joint alignment) | `joint`, `joint-pretrained --model ...` |
| Table 36 (failure prediction) | `failure --model pythia-1.4b` |
| Attention analysis (F.8) | `attention --model ...` |

</details>

**Pre-trained models** (`--model`): `pythia-160m`, `pythia-410m`, `pythia-1.4b`,
`pythia-2.8b`, `pythia-6.9b` (revision `step143000`), `llama-3.2-1b`, `llama-3.2-3b`,
`llama-3.1-8b`, `llama-3.1-8b-instruct`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`,
`gemma-2-2b`. Llama and Gemma require accepting their licenses on Hugging Face.

**Hardware.** The paper's runs used NVIDIA A100 GPUs (40 GB), about 6,480 GPU hours in
total, with Python 3.10.13, PyTorch 2.1.2 (CUDA 12.1), `transformers` 4.44.2,
`tokenizers` 0.19.1, NumPy 1.26.4, SciPy 1.11.4 and scikit-learn 1.3.2, and with TF32
and cuDNN autotuning disabled.

## 📐 Protocol details

The defaults in the code follow the paper's appendices; expand a section for the settings.

<details>
<summary><b>Data</b> (Appendix B)</summary>

- Agreement: 50,000 / 5,000 / 10,000 sentences (train / validation / test); NPI, binding
  and concord: 40,000 / 4,000 / 8,000 each. Labels are balanced; agreement and binding
  are spread evenly over structural loads 0–8.
- Generation seeds: 42 (agreement, binding, concord), 20270723 (scope-sensitive NPI),
  20270722 (separation suite).
- Splits are made by sentence family, so no sentence or minimal pair occurs in two
  splits. When a stratum has fewer distinct families than its quota, records are drawn
  with replacement inside the split.
- DAS pairs come from fresh families: at least 5,000 pairs per load level, split 60/20/20
  by family (3,000 / 1,000 / 1,000), with balanced counterfactual labels computed by the
  SCM. Half of the pairs edit the critical token so that the variable changes; the other
  half use a neutral edit that keeps the variable and the output, so an intervention
  without effect scores 0.5 (`experiments/scratch.py: UNCHANGED_FRACTION`).
- Pre-trained models use 6,000 / 2,000 / 2,000 pairs per variable from 40 templates and
  keep only bases and sources the model handles correctly.

</details>

<details>
<summary><b>From-scratch models</b> (Appendix C.1)</summary>

- Two layers, width 128.
- Transformer: causal pre-LN, 4 heads, FFN 512, GELU, learned positions, Xavier-uniform
  weights, N(0, 0.02²) embeddings. LSTM: unidirectional. Mamba-1: state 16, expansion 2,
  convolution 4, dt rank 8, pre-RMSNorm.
- AdamW (lr 1e-3, weight decay 1.0 on all parameters), 500 warm-up steps and cosine decay,
  batch 64, 50,000 steps, gradient clip 1.0, no dropout, checkpoints every 500 steps.
- Model seeds 1103, 2207, 3319, 4421, 5531.
- Parameter counts at width w match Appendix F.7: Transformer 24w² + 229w + 2,
  LSTM 16w² + 153w + 2, Mamba 12w² + 8w⌈w/16⌉ + 3240w + 2.

</details>

<details>
<summary><b>DAS and controls</b> (Appendix C.3)</summary>

- Cayley rotation `R = R₀ (I − Ω)(I + Ω)⁻¹`; Adam (lr 5e-3), 100 epochs, batch 256,
  best development epoch; ranks {1, 2, 4, 8}.
- Sites: every block output of the pre-trained models (4L configurations per variable);
  in the 2-layer models, the first-block output for atomic variables and the residual
  stream entering the last MLP for compositional ones. Interventions act at the position
  where the variable is consumed.
- DAS seeds 701, 809, 907, 1013, 1109 (the first three for the from-scratch models).
- Controls: wrong SCM; random initialization (step 0, or the pre-trained architecture
  rebuilt from its configuration with seeds 1103–5531); shuffled counterfactual labels.
  Competitors and controls reuse the selected site and rank and refit only `R`.

</details>

<details>
<summary><b>Reading out pre-trained models</b> (Appendix C.4)</summary>

- The model is read where it predicts the critical word (verb, NPI, reflexive or noun).
  The intervention position is the last token before that word, or the last subword of a
  named word when an analysis asks for it (such as the German noun for `can_nom`).
- The network's discrete output compares the total log-probability of the word's two
  candidate forms; critical words with more than three subwords are filtered.
- Behavioral accuracy compares sentence log-probabilities: one BOS token for Llama and
  Gemma, none for Pythia and Qwen, no length normalization, ties broken by pair-ID parity.

</details>

<details>
<summary><b>Sparse autoencoders</b> (Appendix C.5)</summary>

- One SAE per seed, phenomenon, site, architecture and expansion ratio; 50,000 updates
  with batch 4,096.
- TopK with K = 32; JumpReLU and Gated with sparsity coefficients 1e-4 and 3e-4 at 8×,
  chosen to match mean L0 = 32.
- Interchange scores keep the base reconstruction error.

</details>

<details>
<summary><b>Statistics</b> (Appendix C.7)</summary>

- Two-sided tests at α = 0.05 unless noted. Independent units are training seeds for
  from-scratch models and DAS initializations for fixed pre-trained checkpoints.
- Fixed resampling seeds 62026, 6022026, 17012026, 4112026, 9302026 and 7112026
  (see `analysis/stats.py`).

</details>

## 📄 License

Released under the [MIT License](LICENSE).
