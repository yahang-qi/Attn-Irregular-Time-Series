import torch
import torch.nn as nn

from core.models.utils import get_activation_fn


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        forward_expansion: int,
        activation: str='gelu',
        p: float=0.1,
    ):
        super(EncoderMLP, self).__init__()
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim*forward_expansion)
        self.fc3 = nn.Linear(embed_dim*forward_expansion, embed_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x