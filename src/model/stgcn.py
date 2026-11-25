from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


class STGCN(nn.Module):
    # spatio temporal graph convolutional network for per cell crime forecasting
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        history_len: int,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        cheb_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # each stconv block applies two unpadded temporal convolutions of size kernel_size
        total_shrink = num_layers * 2 * (kernel_size - 1)
        # make sure the input window is longer than the total temporal shrinkage
        if history_len <= total_shrink:
            raise ValueError(
                f"history_len ({history_len}) must be greater than {total_shrink} "
                f"for num_layers={num_layers} and kernel_size={kernel_size}."
            )

        # spatiotemporal blocks
        layers: list[STConv] = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(
                STConv(
                    num_nodes=num_nodes,
                    in_channels=current_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    K=cheb_k,
                )
            )
            current_channels = out_channels

        self.blocks = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        remaining_steps = history_len - total_shrink
        # project remaining temporal steps per node into a single nonnegative count
        self.output_head = nn.Sequential(
            nn.Linear(remaining_steps * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        # pass input through each spatiotemporal block with dropout
        for block in self.blocks:
            h = block(h, edge_index, edge_weight)
            h = self.dropout(h)

        batch_size, rem_steps, num_nodes, channels = h.shape
        # move node dimension ahead of time and channels so each cell is flattened independently
        node_features = h.permute(0, 2, 1, 3).reshape(
            batch_size, num_nodes, rem_steps * channels
        )
        # predict one raw scalar per cell in the exact input node order
        raw_counts = self.output_head(node_features).squeeze(-1)
        # enforce nonnegative crime count predictions while keeping smooth gradients near zero
        return F.softplus(raw_counts)