import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GCCBlock(nn.Module):

    def __init__(self, hidden_dim, conv_kernel=4, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel,
            groups=hidden_dim,
            padding=conv_kernel - 1
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: [batch_size, seq_len, hidden_dim]
        输出: [batch_size, seq_len, hidden_dim]
        """
        residual = x
        x = self.norm(x)

        x = self.in_proj(x)  # [batch, seq_len, 2*hidden_dim]
        x, gate = x.chunk(2, dim=-1)

        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        x = self.conv1d(x)
        x = x[..., :residual.size(1)]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        x = x * torch.sigmoid(gate)

        x = self.out_proj(x)
        x = self.dropout(x)
        # x = self.out_proj(x)
        return residual + x


class GCCModel(nn.Module):

    def __init__(
            self,
            hops,
            n_class,
            input_dim,
            pe_dim,
            n_layers=2,
            hidden_dim=64,
            dropout_rate=0.0,
            conv_kernel=3#acm:3 dblp:3 academic:8 freebase:3
    ):
        super().__init__()
        self.seq_len = hops + 1
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        print('conv_kernel:', conv_kernel)

        self.layers = nn.ModuleList([
            GCCBlock(
                hidden_dim=hidden_dim,
                conv_kernel=conv_kernel,
                dropout=dropout_rate
            ) for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        self.attn_layer = nn.Linear(2 * hidden_dim, 1)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        """
        input: [batch_size, seq_len, input_dim]
        output: [batch_size, hidden_dim//2]
        """
        x = self.input_proj(batched_data)

        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        target = x[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        node_tensor, neighbor_tensor = torch.split(x, [1, self.seq_len - 1], dim=1)

        attn_weights = F.softmax(
            self.attn_layer(torch.cat((target, neighbor_tensor), dim=2)),
            dim=1
        )

        neighbor_tensor = torch.sum(neighbor_tensor * attn_weights, dim=1, keepdim=True)
        output = (node_tensor + neighbor_tensor).squeeze(1)

        output = F.relu(self.out_proj(output))
        return output
