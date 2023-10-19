import torch


def sinusoidal_embedding(n, d):
    embedding = torch.tensor([[i / 1e4 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[1 - sin_mask])

    return embedding
