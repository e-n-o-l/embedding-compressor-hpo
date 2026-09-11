from torch.nn import Module, Linear
from .cross_attention import CrossAttention
from torch import Tensor

class OfferCompressor(Module):
    def __init__(self, input_dim: int, target_dim: int, num_clusters: int, num_heads: int):
        super().__init__()

        self.linear = Linear(in_features=input_dim, out_features=target_dim)

        self.attention = CrossAttention(
            num_heads=num_heads,
            embed_dim=target_dim,
            seq_len=num_clusters
        )

    def forward(self, x: Tensor, padding_mask: Tensor = None):
        x_comp = self.linear(x)
        centroids = self.attention(x_comp, padding_mask=padding_mask)
        return x_comp, centroids