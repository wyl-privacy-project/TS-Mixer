import torch
from torch import nn
class TS_Mixer(nn.Module):

    def __init__(self, num_mixers: int, max_seq_len: int, hidden_dim: int, mlp_hidden_dim: int, **kwargs):
        super(TS_Mixer, self).__init__(**kwargs)
        self.hidden_dim=hidden_dim
        self.max_len=max_seq_len
        self.mixers = nn.Sequential(*[
            MixerLayer(max_seq_len, hidden_dim, mlp_hidden_dim) for _ in range(num_mixers)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mixers(inputs)

def compute(len_seq):
    if len_seq == 32:
        cycle_num = 3
    elif len_seq == 64:
        cycle_num = 4
    elif len_seq >= 128 and len_seq <=1024:
        cycle_num = 5
    else:
        cycle_num = 6
    return cycle_num

class MixerLayer(nn.Module):

    def __init__(self, max_seq_len: int, hidden_dim: int, channel_hidden_dim: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.FC1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.FC2_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.FC2_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.FC3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MlpLayer(hidden_dim, channel_hidden_dim)


    def forward(self, x):
        B, N, D = x.shape
        x1 = self.FC1(x)
        x_shift1 = torch.zeros(B, N,D)
        x_shift2 = torch.zeros(B, N, D)
        cycle_num = compute(self.max_seq_len)
        for i in range(6):
            x_shift1 += torch.roll(x1, (0, 0, i+1), (1, 1, 1))
            x_shift2 += torch.roll(x1, (0, 0, -i-1), (1, 1, 1))
        x_shift1 = self.FC2_1(x_shift1)
        x_shift2 = self.FC2_2(x_shift2)
        outputs = self.FC3(x_shift1+x_shift2)
        outputs = (outputs+x1)#+outputs2
        # 以下为 channel mixing 不改变
        residual = outputs#BND
        outputs = self.layer_norm(residual)
        channel_outputs = self.mlp(outputs)
        outputs = (channel_outputs + residual)
        return outputs

class MlpLayer(nn.Module):

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        ])
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

