"""Specification-aligned progress measure (Section 5.2).

Computes per-variable IIA at every 500th training step to track
which prescribed variable the network is learning to represent.
"""
import torch
from src.das.das_trainer import DASTrainer

def compute_progress_trajectory(checkpoints, das_pairs, layer, target_pos,
                                 das_config, step_interval=500):
    """Compute IIA at each checkpoint for one variable."""
    trajectory = []
    for step, model in checkpoints:
        trainer = DASTrainer(model, **das_config)
        iia = trainer.train(das_pairs, layer, target_pos)
        trajectory.append({"step": step, "iia": float(iia)})
    return trajectory
