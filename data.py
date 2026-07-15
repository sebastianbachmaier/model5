"""
data.py — memory-mapped reader for the flat uint16 token shards produced by
data_prep.py. Yields fixed-length (x, y) windows where y = x shifted by one
token, with each DDP rank reading a disjoint stride of the global window
index space.
"""
import glob
import json
import os

import numpy as np
import torch


class TokenDataset:
    """Sequential, memory-mapped reader over one or more .bin token shards.

    The whole set of shards is treated as one virtual concatenated token
    stream of length `total_tokens`, split into non-overlapping windows of
    length `seq_len + 1` (the extra token gives us the shifted target).
    Windows are indexed globally as `window_idx = 0, 1, 2, ...`; rank r only
    ever reads windows where `window_idx % world_size == r`, guaranteeing
    ranks never train on the same data in the same step.
    """

    def __init__(self, data_dir: str, seq_len: int, rank: int = 0, world_size: int = 1):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        shard_paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        assert len(shard_paths) > 0, f"no .bin shards found in {data_dir}"
        self.shard_paths = shard_paths
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        self.shard_sizes = [len(s) for s in self.shards]
        self.shard_starts = np.cumsum([0] + self.shard_sizes[:-1])
        self.total_tokens = int(sum(self.shard_sizes))
        self.num_windows = self.total_tokens // seq_len

        meta_path = os.path.join(data_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None

        # cursor = index of the next window this rank will read. Starts at
        # `rank` so ranks are staggered from step 0.
        self.cursor = rank

    def set_position(self, num_windows_consumed_per_rank: int):
        """Fast-forward the cursor to resume training from a given step,
        e.g. `num_windows_consumed_per_rank = step * micro_bsz * grad_accum`."""
        self.cursor = self.rank + num_windows_consumed_per_rank * self.world_size

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens starting at global token offset `start`,
        wrapping around to the beginning of the dataset if needed (only
        relevant once training has consumed the whole corpus more than
        once)."""
        out = np.empty(length, dtype=np.uint16)
        pos = 0
        idx = start % self.total_tokens
        while pos < length:
            shard_i = int(np.searchsorted(self.shard_starts, idx, side="right") - 1)
            local_off = idx - self.shard_starts[shard_i]
            shard = self.shards[shard_i]
            avail = len(shard) - local_off
            take = min(avail, length - pos)
            out[pos:pos + take] = shard[local_off:local_off + take]
            pos += take
            idx = (idx + take) % self.total_tokens
        return out

    def _get_window(self, window_idx: int):
        start = (window_idx % self.num_windows) * self.seq_len
        tokens = self._read_tokens(start, self.seq_len + 1)
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        return x, y

    def next_batch(self, batch_size: int, device):
        xs, ys = [], []
        for _ in range(batch_size):
            x, y = self._get_window(self.cursor)
            xs.append(x)
            ys.append(y)
            self.cursor += self.world_size
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        return x, y
