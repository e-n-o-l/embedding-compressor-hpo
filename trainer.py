import torch
from compressor_losses import ComposedLoss
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.functional import normalize
from torchmetrics.retrieval import (RetrievalMAP, RetrievalMRR, RetrievalHitRate,
                                    RetrievalPrecision, RetrievalRecall, RetrievalNormalizedDCG)


def train_compressor(model: Module, epochs: int,
                     train_loader: DataLoader,
                     lambda_cluster: float,
                     device: torch.device = torch.device("cpu")
) -> Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ComposedLoss(lambda_cluster=lambda_cluster).to(device)

    model.train()

    for epoch in range(epochs):
        for x_orig, padding_mask, _, _ in train_loader:

            x_orig = x_orig.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            x_comp, centroids = model(x_orig, padding_mask=padding_mask)

            loss = criterion(x_orig, x_comp, centroids, padding_mask)

            loss.backward()
            optimizer.step()

    return model


def evaluate_compressor(model: Module, val_loader: DataLoader, k: int = 5,
                        device: torch.device = torch.device("cpu")
) -> dict:
    model = model.to(device)
    model.eval()

    all_centroids = []
    all_metadata = []

    with torch.no_grad():

        for x_orig, padding_mask, _, metadata in val_loader:

            x_orig = x_orig.to(device)

            padding_mask = padding_mask.to(device)

            _, centroids = model(x_orig, padding_mask=padding_mask)

            for i in range(centroids.size(0)):

                all_centroids.append(centroids[i])

                all_metadata.extend([metadata[i]] * centroids.size(1))

    all_centroids = torch.cat(all_centroids, dim=0).to(device)

    all_centroids = normalize(all_centroids, p=2, dim=-1)

    sim_matrix = torch.mm(all_centroids, all_centroids.T)

    n = sim_matrix.size(0)

    unique_meta = {meta: idx for idx, meta in enumerate(set(all_metadata))}
    meta_ids = [unique_meta[m] for m in all_metadata]

    meta_tensor = torch.tensor(meta_ids, device=device)
    targets = (meta_tensor.unsqueeze(0) == meta_tensor.unsqueeze(1))
    targets.fill_diagonal_(False)

    targets = targets.to(device)


    indexes = torch.arange(n, device=device).repeat_interleave(n).long()

    pred_flat = sim_matrix.view(-1)

    targets_flat = targets.view(-1)


    recall = RetrievalRecall(top_k=k).to(device)

    precision = RetrievalPrecision(top_k=k).to(device)

    mrr = RetrievalMRR().to(device)

    hit = RetrievalHitRate(top_k=k).to(device)

    rmap = RetrievalMAP().to(device)

    normalized_dcg = RetrievalNormalizedDCG(top_k=k).to(device)

    return {
        "recall_k": recall(pred_flat, targets_flat, indexes=indexes).item(),
        "mrr": mrr(pred_flat, targets_flat, indexes=indexes).item(),
        "precision_k": precision(pred_flat, targets_flat, indexes=indexes).item(),
        "hit": hit(pred_flat, targets_flat, indexes=indexes).item(),
        "map": rmap(pred_flat, targets_flat, indexes=indexes).item(),
        "normalized_dcg": normalized_dcg(pred_flat, targets_flat, indexes=indexes).item(),
    }