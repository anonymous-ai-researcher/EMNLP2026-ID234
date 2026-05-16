"""Negative control experiments (Section 4, Appendix D.6).

Three controls to validate that high IIA reflects genuine causal alignment:
1. Wrong SCM: apply DAS using a different phenomenon's SCM (12 off-diagonal pairings)
2. Random initialization: DAS on an untrained, randomly initialized network
3. Shuffled labels: randomly permute the source variable labels
"""

def run_wrong_scm_control(model, phenomenon, layer, target_pos, das_config):
    """Run DAS with wrong SCM (e.g., agreement SCM on NPI data)."""
    all_phenomena = ["agreement", "npi", "binding", "concord"]
    wrong_phenomena = [p for p in all_phenomena if p != phenomenon]
    results = {}
    for wrong_phen in wrong_phenomena:
        # Run DAS with wrong_phen's variable definitions on phenomenon's data
        pass  # IIA should be ~0.50
    return results

def run_random_init_control(model_class, das_config):
    """Run DAS on a randomly initialized (untrained) network."""
    pass  # IIA should be ~0.50

def run_shuffled_labels_control(model, pairs, layer, target_pos, das_config):
    """Run DAS with randomly permuted source labels."""
    import numpy as np
    # Shuffle the counterfactual labels
    pass  # IIA should be ~0.50
