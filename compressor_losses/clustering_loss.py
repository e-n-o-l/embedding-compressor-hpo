from torch.nn import Module
from torch.nn.functional import normalize
import torch


class ClusteringLoss(Module):
    def __init__(self):
        super(ClusteringLoss, self).__init__()

    def forward(self, x_comp, centroids, padding_mask):
        centroids_norm = normalize(centroids, p=2, dim=-1)

        cluster_losses = []

        for i in range(x_comp.size(0)):
            offer_mask = ~padding_mask[i]
            offer_photos = normalize(x_comp[i][offer_mask], p=2, dim=-1)
            offer_centroids = centroids_norm[i]

            sim_matrix = torch.mm(offer_photos, offer_centroids.T)

            max_sim_per_photo, _ = torch.max(sim_matrix, dim=-1)

            offer_loss = 1.0 - max_sim_per_photo.mean()
            cluster_losses.append(offer_loss)

        return torch.stack(cluster_losses).mean()