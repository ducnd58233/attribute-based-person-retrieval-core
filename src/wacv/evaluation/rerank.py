"""Query-set normalisation of a distance matrix."""

from __future__ import annotations

import numpy as np


def dual_softmax_rerank(
    distances: np.ndarray,
    *,
    tau: float = 0.05,
    chunk_cols: int = 4096,
) -> np.ndarray:
    """Query-set dual softmax over gallery columns (no labels, no updates).

    Converts distances ``D`` to similarities ``S = -D``, then
    ``S' = S + log_softmax_{queries}(S / tau)`` and returns ``-S'`` as a
    distance. Softmax is column-wise (over queries for each gallery image),
    which dampens hubness without using ground-truth matches.

    Gallery columns are processed in chunks so peak RAM stays bounded on
    Codabench-sized Q×G matrices (float64 full-matrix form peaked ~6GB).

    Works in place to avoid a second full Q x G array: a float32 input is
    overwritten and returned (any other dtype is cast to a float32 copy
    first). Mirrors submission/run.py's copy of this function. Both call sites
    reassign their ``distances`` to the return value and never read the old
    array again; a caller that needs the input kept should pass a copy.
    """
    similarities = np.asarray(distances, dtype=np.float32)
    np.negative(similarities, out=similarities)
    scale = max(float(tau), 1e-8)
    n_gallery = similarities.shape[1]
    step = max(int(chunk_cols), 1)
    for start in range(0, n_gallery, step):
        end = min(start + step, n_gallery)
        block = similarities[:, start:end] / scale
        col_max = block.max(axis=0, keepdims=True)
        exp_shifted = np.exp(block - col_max)
        log_partition = np.log(
            np.maximum(exp_shifted.sum(axis=0, keepdims=True), 1e-12)
        )
        log_softmax = block - col_max - log_partition
        similarities[:, start:end] = -(similarities[:, start:end] + log_softmax)
    return similarities
