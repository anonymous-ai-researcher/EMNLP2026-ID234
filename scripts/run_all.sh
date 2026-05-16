#!/bin/bash
# Master script: reproduces all results from the paper.
# Estimated runtime: ~420 GPU-hours on NVIDIA A100 40GB.
set -e

echo "=== Step 1: Generate data ==="
python scripts/train_from_scratch.py --stage data --seed 42

echo "=== Step 2: Train from-scratch models ==="
for arch in transformer lstm mamba; do
    for seed in 1 2 3 4 5; do
        for phen in agreement npi binding concord; do
            python scripts/train_from_scratch.py \
                --config configs/${arch}.yaml \
                --phenomenon $phen --seed $seed
        done
    done
done

echo "=== Step 3: DAS verification (from-scratch) ==="
python scripts/run_das.py --model_dir checkpoints/ --config configs/das.yaml

echo "=== Step 4: DAS verification (pre-trained) ==="
for model in EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1.4b \
             meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B; do
    python scripts/run_das_pretrained.py --model_name $model --config configs/das.yaml
done

echo "=== Step 5: SAE analysis ==="
python scripts/run_sae.py --model_dir checkpoints/transformer/ --config configs/das.yaml

echo "=== Step 6: Ablations & controls ==="
python scripts/run_ablations.py --all
python scripts/run_negative_controls.py --all

echo "=== Step 7: Generate figures ==="
cd figures && for f in gen_fig_*.py; do python $f; done && cd ..

echo "=== All experiments complete. ==="
