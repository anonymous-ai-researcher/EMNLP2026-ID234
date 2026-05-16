## From Specification to Causal Verification: Do Transformers Implement Formally Prescribed Algorithms for Syntax?

> **Anonymous submission to EMNLP 2026 (ARR May 2026, Paper #234)**

---

## Overview

This repository provides the complete codebase for reproducing all experiments, figures, and tables in the paper. We bridge formal language theory (B-RASP) and causal mechanistic interpretability (DAS/IIA) to verify whether Transformers implement formally prescribed algorithms for four English syntactic phenomena.

```
Syntactic Constraint  →  B-RASP Program  →  Structural Causal Model  →  Causal Verification (DAS)
```

### Key Results

| Finding | Evidence |
|---|---|
| From-scratch Transformers achieve IIA ≥ 0.95 on all 4 specifications | Table 3 |
| Prescribed variables emerge during grokking in causal-dependency order | Figure 3 |
| SAE features fail on compositional variables; DAS succeeds | Table 4 |
| Transformer-Mamba succinctness gap on agreement | Figure 4 |
| Pre-trained models (up to 3B) show strong but imperfect alignment (0.82-0.95) | Table 3 |

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── configs/
│   ├── transformer.yaml        # From-scratch Transformer hyperparameters
│   ├── lstm.yaml               # LSTM hyperparameters
│   ├── mamba.yaml              # Mamba hyperparameters
│   └── das.yaml                # DAS training configuration
├── src/
│   ├── data/
│   │   ├── cfg_generator.py    # CFG-based sentence generation (§4, App B.1)
│   │   ├── lexicon.py          # Lexicon definitions (Table 5)
│   │   ├── blimp_loader.py     # BLiMP paradigm loader (App B.2)
│   │   └── das_pairs.py        # DAS counterfactual pair construction (App B.3)
│   ├── models/
│   │   ├── transformer.py      # 2-layer Transformer (d=128, H=4)
│   │   ├── lstm.py             # LSTM baseline (matched params)
│   │   ├── mamba.py            # Mamba baseline (matched params)
│   │   └── pretrained.py       # Pre-trained model loader (Pythia, Llama)
│   ├── das/
│   │   ├── das_trainer.py      # DAS optimization with Cayley parameterization
│   │   ├── rotation.py         # Orthogonal rotation (Cayley transform)
│   │   ├── iia.py              # IIA computation (Eq. 1)
│   │   └── negative_controls.py # Wrong-SCM, random-init, shuffled-label controls
│   ├── sae/
│   │   ├── sae_model.py        # TopK Sparse Autoencoder
│   │   ├── sae_trainer.py      # SAE training loop
│   │   └── alignment.py        # SAE feature alignment scoring
│   ├── evaluation/
│   │   ├── behavioral.py       # Minimal-pair accuracy evaluation
│   │   ├── blimp_eval.py       # BLiMP log-probability evaluation
│   │   └── progress_measure.py # Specification-aligned progress measure (§5.2)
│   └── utils/
│       ├── scm.py              # SCM definitions for 4 phenomena
│       ├── brasp.py            # B-RASP program definitions
│       └── stats.py            # Statistical testing (paired t-test, bootstrap)
├── scripts/
│   ├── train_from_scratch.py   # Train from-scratch models (Table 3, left)
│   ├── run_das.py              # Run DAS verification (Table 3)
│   ├── run_das_pretrained.py   # DAS on pre-trained models (Table 3, right)
│   ├── run_sae.py              # SAE analysis (Table 4)
│   ├── run_ablations.py        # All ablation experiments (Appendix D)
│   ├── run_negative_controls.py # Negative control experiments
│   └── run_all.sh              # Master script: reproduces all results
├── figures/
│   ├── gen_fig_pipeline.py     # Figure 1: Pipeline overview
│   ├── gen_fig_perlayer.py     # Figure 2: Per-layer IIA
│   ├── gen_fig_trajectory.py   # Figure 3: IIA trajectory during grokking
│   ├── gen_fig_succinctness.py # Figure 4: Succinctness gap
│   ├── gen_fig_comparison.py   # Figure 5: Trajectory comparison (App E.2)
│   └── gen_fig_scaling.py      # Figure 6: IIA scaling (App E.4)
└── paper/
    ├── emnlp26-id234.tex       # LaTeX source
    ├── emnlp.bib               # Bibliography
    └── figures/                # Compiled figure PDFs
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n brasp-verify python=3.10
conda activate brasp-verify
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Generate CFG-based training data for all 4 phenomena (seed=42)
python scripts/train_from_scratch.py --stage data --seed 42

# Download BLiMP paradigms for pre-trained evaluation
python src/data/blimp_loader.py --output data/blimp/
```

### 3. Train From-Scratch Models

```bash
# Train 2-layer Transformer (5 seeds × 4 phenomena)
for seed in 1 2 3 4 5; do
    for phen in agreement npi binding concord; do
        python scripts/train_from_scratch.py \
            --config configs/transformer.yaml \
            --phenomenon $phen \
            --seed $seed \
            --max_steps 50000 \
            --checkpoint_every 500
    done
done

# Train LSTM and Mamba baselines (matched parameters)
python scripts/train_from_scratch.py --config configs/lstm.yaml --all
python scripts/train_from_scratch.py --config configs/mamba.yaml --all
```

### 4. Run DAS Verification

```bash
# DAS on from-scratch models (rank search over {1,2,4,8})
python scripts/run_das.py \
    --model_dir checkpoints/transformer/ \
    --config configs/das.yaml \
    --ranks 1 2 4 8

# DAS on pre-trained models (3 random initializations)
for model in EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1.4b \
             meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B; do
    python scripts/run_das_pretrained.py \
        --model_name $model \
        --config configs/das.yaml \
        --n_inits 3
done
```

### 5. Run SAE Analysis

```bash
# Train SAEs on from-scratch Transformer residual streams
python scripts/run_sae.py \
    --model_dir checkpoints/transformer/ \
    --expansion 8 \
    --topk 32 \
    --steps 50000
```

### 6. Run All Ablations

```bash
# Rank ablation (Table 9), per-layer (Table 10), distance (Table 11),
# alt-SCM (Table 12), SAE size (Table 13), negative controls (Table 14),
# succinctness (Table 15), Sutter replication (Table 16)
python scripts/run_ablations.py --all
```

### 7. Reproduce All (Single Command)

```bash
bash scripts/run_all.sh  # ~420 GPU-hours on A100 40GB
```

---

## Reproducibility Details

### From-Scratch Models (Appendix C.1)

| Hyperparameter | Transformer | LSTM | Mamba |
|---|---|---|---|
| Layers | 2 | 2 | 2 |
| Hidden dim $d$ | 128 | 128 | 128 |
| Heads $H$ | 4 | -- | -- |
| State dim | -- | -- | 16 |
| Expand factor | -- | -- | 2 |
| Parameters | ~1.05M | ~1.08M | ~1.15M |
| Activation | GELU | tanh | SiLU |
| Positional encoding | Learned absolute | -- | -- |
| Initialization | Xavier uniform | Xavier uniform | Xavier uniform |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1.0 | 1.0 | 1.0 |
| Warmup steps | 500 | 500 | 500 |
| LR schedule | Cosine | Cosine | Cosine |
| Max steps | 50,000 | 50,000 | 50,000 |
| Batch size | 64 | 64 | 64 |
| Dropout | 0.0 | 0.0 | 0.0 |
| Vocab size | 135 | 135 | 135 |
| Max sequence length | 25 | 25 | 25 |
| Loss | Cross-entropy | Cross-entropy | Cross-entropy |
| Seeds | 5 | 5 | 5 |

### DAS Configuration (Appendix C.3)

| Parameter | Value |
|---|---|
| Rotation parameterization | Cayley transform |
| Optimizer | Adam |
| Learning rate | 5e-3 |
| Epochs | 100 |
| Batch size | 256 |
| Validation split | 20% |
| Rank search | {1, 2, 4, 8} |
| Layer search | All layers |
| Checkpoint selection | Highest validation IIA |
| Token position | Consumed position |
| Intervention hook | After full block (residual stream) |

### SAE Configuration (Appendix C.5)

| Parameter | Value |
|---|---|
| Architecture | TopK |
| Expansion ratio $E$ | 8 (dictionary size 1024) |
| $K$ | 32 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 4096 |
| Training steps | 50,000 |
| Reconstruction MSE | ≤ 0.005 |
| Dead features | < 5% |

### Compute Budget (~420 GPU-hours on NVIDIA A100 40GB)

| Component | GPU-hours |
|---|---|
| From-scratch training (3 arch × 4 phen × 5 seeds) | 120 |
| DAS from-scratch (8 var × 8 cfg × 5 seeds) | 40 |
| DAS pre-trained (8 var × 96 cfg × 3 runs) | 200 |
| SAE (2 layers × 4 phen × 5 seeds) | 20 |
| Evaluation + ablations | 40 |
| **Total** | **~420** |

---

## Software

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1 |
| CUDA | 12.1 |
| transformers | 4.36+ |
| mamba-ssm | 1.2+ |

---

## Figures

All figures can be regenerated from the `figures/` directory:

```bash
cd figures
python gen_fig_pipeline.py      # Figure 1
python gen_fig_perlayer.py      # Figure 2
python gen_fig_trajectory.py    # Figure 3
python gen_fig_succinctness.py  # Figure 4
python gen_fig_comparison.py    # Figure 5
python gen_fig_scaling.py       # Figure 6
```

Color scheme (uniform across all figures):
- Blue `#2166AC`: atomic variables / Transformer
- Red `#B2182B`: compositional variables / Mamba
- Green `#4DAF4A`: LSTM

---

## License

This code is released under the MIT License for research purposes.
