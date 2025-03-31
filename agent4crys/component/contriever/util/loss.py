import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        # self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, z_i, z_j):
        n = z_i.size(0)
        negatives_mask = (
            ~torch.eye(n * 2, n * 2, dtype=bool, device=z_i.device)
        ).float()

        z_i = F.normalize(z_i, dim=1)  # (batch_size, dim)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)  # (batch_size*2, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # (batch_size*2, batch_size*2)

        sim_ij = torch.diag(similarity_matrix, n)
        sim_ji = torch.diag(similarity_matrix, -n)  # (batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (batch_size * 2)
        nominator = torch.exp(positives / self.temperature)

        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * n)
        return loss
