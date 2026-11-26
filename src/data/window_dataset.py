from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_grid_graph(
    grid_coords: np.ndarray,
    centroids: np.ndarray | None = None,
    distance_weighted: bool = False,
    sigma_km: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # build undirected edges for grid cells and weight them by distance
    num_nodes = len(grid_coords)
    # map each row and col coordinate pair to its node index
    coord_to_node = {
        (int(row), int(col)): idx for idx, (row, col) in enumerate(grid_coords)
    }

    edges: list[tuple[int, int]] = []
    for idx, (row, col) in enumerate(grid_coords):
        # check the four cardinal neighbors around each cell
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (int(row) + d_row, int(col) + d_col)
            if neighbor in coord_to_node:
                # add both directions so message passing works symmetrically
                edges.append((idx, coord_to_node[neighbor]))

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if distance_weighted:
        if centroids is None:
            raise ValueError("centroids are required when distance_weighted is true")
        # compute euclidean distance between cell centroids if distance weights are requested
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        dists = np.linalg.norm(centroids[src] - centroids[dst], axis=1)
        # apply a gaussian kernel so closer cells get higher edge weights
        weights = np.exp(-(dists ** 2) / (2.0 * (sigma_km ** 2)))
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        # fall back to uniform weights of one if no distance scaling is used
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    if edge_index.numel() > 0:
        assert edge_index.min() >= 0 and edge_index.max() < num_nodes
    return edge_index, edge_weight


@dataclass
class WindowConfig:
    # holds split ratios and window dimensions for building datasets
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    history_len: int = 14
    horizon: int = 1
    distance_weighted: bool = False
    sigma_km: float = 2.0


class CrimeWindowDataModule:
    # load the processed dataset and create chronological splits
    def __init__(
        self,
        npz_path: str | Path,
        meta_path: str | Path,
        cfg: WindowConfig,
    ):
        self.cfg = cfg
        data = np.load(npz_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        x_raw = data["x"].astype(np.float32)
        # keep targets in raw count space so evaluation reflects actual crime counts
        self.y_raw = data["y"].astype(np.float32)
        self.cell_ids: list[str] = list(self.meta["cell_ids"])
        self.dates: list[str] = list(self.meta["dates"])

        # use provided graph edges or build them from grid coordinates
        if "edge_index" in data:
            self.edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long)
            if "edge_weight" in data:
                self.edge_weight = torch.as_tensor(data["edge_weight"], dtype=torch.float32)
            else:
                self.edge_weight = torch.ones(self.edge_index.shape[1], dtype=torch.float32)
        else:
            grid_coords = np.asarray(self.meta["grid_coords"], dtype=np.int64)
            centroids = (
                np.asarray(self.meta["cell_centroids"], dtype=np.float32)
                if "cell_centroids" in self.meta
                else None
            )
            self.edge_index, self.edge_weight = build_grid_graph(
                grid_coords=grid_coords,
                centroids=centroids,
                distance_weighted=cfg.distance_weighted,
                sigma_km=cfg.sigma_km,
            )

        # compute chronological cutoff indices before creating any windows
        total_days = x_raw.shape[0]
        self.train_end = int(total_days * cfg.train_ratio)
        self.val_end = int(total_days * (cfg.train_ratio + cfg.val_ratio))

        # fit feature normalization strictly on the training slice
        train_slice = x_raw[: self.train_end]
        self.feat_mean = train_slice.mean(axis=(0, 1), keepdims=True)
        raw_std = train_slice.std(axis=(0, 1), keepdims=True)
        # replace zero standard deviations with one to prevent division by zero
        self.feat_std = np.where(raw_std < 1e-8, 1.0, raw_std)

        self.x_scaled = ((x_raw - self.feat_mean) / self.feat_std).astype(np.float32)

    def get_split(self, split: str) -> "CrimeWindowDataset":
        # return a windowed dataset isolated to a single chronological split
        bounds = {
            "train": (0, self.train_end),
            "val": (self.train_end, self.val_end),
            "test": (self.val_end, self.x_scaled.shape[0]),
        }
        if split not in bounds:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

        start, end = bounds[split]
        return CrimeWindowDataset(
            x=self.x_scaled[start:end],
            y=self.y_raw[start:end],
            history_len=self.cfg.history_len,
            horizon=self.cfg.horizon,
            dates=self.dates[start:end],
        )


class CrimeWindowDataset(Dataset):
    # build history windows and horizon targets in one split
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        history_len: int,
        horizon: int,
        dates: list[str],
    ):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.history_len = history_len
        self.horizon = horizon

        # only keep forecast origins whose full lookback window and target stay inside this slice
        self.origins = list(range(history_len - 1, len(x) - horizon))
        self.origin_dates = [dates[t] for t in self.origins]
        self.target_dates = [dates[t + horizon] for t in self.origins]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.origins[idx]
        # grab history_len days up to and including forecast origin t
        x_win = self.x[t - self.history_len + 1 : t + 1]
        # grab raw target crime counts at t plus horizon
        y_target = self.y[t + self.horizon]
        return x_win, y_target
