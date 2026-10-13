#!/usr/bin/env bash
# Full reproduction in dependency order (GPU strongly recommended).
set -euo pipefail
run() { python -m spec2verify "$@"; }

# from-scratch models (Section 5, Appendices C-F)
for arch in transformer lstm mamba; do run table2-scratch --arch "$arch"; done
run structure-english
run structure-german
for regime in label_only aux label_input; do run trajectories --regime "$regime"; done
run excluded-loss
run sae
run boolean
run rank-ablation
run rank-by-layer
run three-input
run distance-alternatives
run transfer
run concord-stress
run unconstrained
run probing
run joint

# size sweep (Section 5.5)
run arch-tune
for arch in transformer lstm mamba; do run arch-sweep --arch "$arch"; run arch-growth --arch "$arch"; done
run mamba-probe

# pre-trained models
for m in pythia-160m pythia-410m pythia-1.4b llama-3.2-1b llama-3.2-3b; do
  run table2-pretrained --model "$m"
  run wrong-scm-pretrained --model "$m"
  run blimp --model "$m"
  run structure-english-pretrained --model "$m"
  run joint-pretrained --model "$m"
done
for m in pythia-2.8b pythia-6.9b llama-3.1-8b llama-3.1-8b-instruct; do
  run table2-pretrained --model "$m"
  run structure-english-pretrained --model "$m"
done
for m in llama-3.2-1b llama-3.2-3b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b gemma-2-2b; do
  run structure-german-pretrained --model "$m"
done
run failure --model pythia-1.4b
for m in pythia-1.4b llama-3.2-3b; do run attention --model "$m"; done

run figures
