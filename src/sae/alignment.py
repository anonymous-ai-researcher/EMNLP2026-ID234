"""SAE feature alignment scoring (Section 5.3, Appendix C.5).

Al(f, U) = max(corr(f_activations, U_values), corr(f_activations, 1-U_values))
"""
import torch
import numpy as np

def compute_alignment(sae_activations, variable_values):
    """Compute alignment between each SAE feature and a prescribed variable."""
    n_features = sae_activations.shape[1]
    alignments = []
    for f in range(n_features):
        feat = sae_activations[:, f].numpy()
        var = variable_values.numpy().astype(float)
        corr_pos = np.corrcoef(feat, var)[0, 1]
        corr_neg = np.corrcoef(feat, 1 - var)[0, 1]
        alignments.append(max(abs(corr_pos), abs(corr_neg)))
    return np.array(alignments)

def best_feature_iia(alignments, threshold=0.8):
    """Find the best-aligned feature and compute its effective IIA."""
    best_idx = np.argmax(alignments)
    return best_idx, alignments[best_idx]
