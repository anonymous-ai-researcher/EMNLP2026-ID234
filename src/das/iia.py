"""Interchange Intervention Accuracy computation (Equation 1).

IIA(U) = E_{b,s} [ N(do(h_i^l := R R^T h_i^l(s) + (I - R R^T) h_i^l(b))) == C(do(U := u_s)) ]
"""
import torch

def compute_iia(model, rotation, base_inputs, source_inputs, base_labels,
                counterfactual_labels, layer, target_pos):
    """Compute IIA for a single variable."""
    R = rotation()
    proj = R @ R.T
    
    with torch.no_grad():
        h_base = model.get_residual_stream(base_inputs, layer, target_pos)
        h_source = model.get_residual_stream(source_inputs, layer, target_pos)
    
    h_intervened = proj @ h_source.unsqueeze(-1) + (torch.eye(proj.shape[0], device=proj.device) - proj) @ h_base.unsqueeze(-1)
    h_intervened = h_intervened.squeeze(-1)
    
    # Forward pass with intervened activations
    logits = model.classifier(h_intervened)
    preds = logits.argmax(dim=-1)
    iia = (preds == counterfactual_labels).float().mean()
    return iia
