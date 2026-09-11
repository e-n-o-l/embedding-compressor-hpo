from torch.nn import Module
import torch
from torch.nn.functional import normalize, mse_loss

class DimensionCompressorLoss(Module):

    def __init__(self):
        super(DimensionCompressorLoss, self).__init__()


    def forward(self, x_orig, x_comp, padding_mask):

        real_indices = ~padding_mask
        flat_orig = x_orig[real_indices]
        flat_comp = x_comp[real_indices]

        with torch.no_grad():
            orig_norm = normalize(flat_orig, p=2, dim=-1)
            target_sim = torch.mm(orig_norm, orig_norm.T)

        comp_norm = normalize(flat_comp, p=2, dim=-1)
        pred_sim = torch.mm(comp_norm, comp_norm.T)

        return mse_loss(pred_sim, target_sim)