import torch
from torch.utils.data import Dataset
import numpy as np


class CarOfferDataset(Dataset):
    def __init__(self, embeddings, urls, metadata):
        self.urls = np.array(urls)
        self.unique_urls = np.unique(self.urls)
        self.embeddings = embeddings
        self.metadata = np.array(metadata)

        self.grouped_indices = {url: np.where(self.urls == url)[0] for url in self.unique_urls}

    def __len__(self):
        return len(self.unique_urls)

    def __getitem__(self, idx):
        url = self.unique_urls[idx]
        indices = self.grouped_indices[url]

        offer_embeds = self.embeddings[indices]
        offer_meta = self.metadata[indices[0]]

        return torch.tensor(offer_embeds, dtype=torch.float32), url, offer_meta