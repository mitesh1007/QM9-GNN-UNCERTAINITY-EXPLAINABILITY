import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """GCN for QM9 HOMO-LUMO gap prediction."""

    def __init__(
        self,
        input_dim=11,
        hidden_dim=64,
        output_dim=1,
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Convert node representations into
        # one representation for each molecule.
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x