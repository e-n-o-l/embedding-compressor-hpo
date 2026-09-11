from torch.nn import Module
from .clustering_loss import ClusteringLoss
from .dim_compressor_loss import DimensionCompressorLoss

class ComposedLoss(Module):

    def __init__(self, lambda_cluster):
        super(ComposedLoss, self).__init__()

        self.lambda_cluster = lambda_cluster

        self.clustering_loss = ClusteringLoss()

        self.dim_loss = DimensionCompressorLoss()


    def forward(self, x_orig, x_comp, centroids, padding_mask):
        d_loss = self.dim_loss(x_orig, x_comp, padding_mask)

        c_loss = self.clustering_loss(x_comp, centroids, padding_mask)

        return c_loss * self.lambda_cluster + d_loss