import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always", nogil=True, boundscheck=False)
def maximum_path_each_numba(
    path: NDArray[np.int32],
    value: NDArray[np.float32],
    t_x: int,
    t_y: int,
    max_neg_val: float = -1e9,
) -> None:
    """Compute maximum monotonic alignment path for one sample."""
    index = t_x - 1

    for y in range(t_y):
        start_x = max(0, t_x + y - t_y)
        end_x = min(t_x, y + 1)
        for x in range(start_x, end_x):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y - 1] if y > 0 else max_neg_val

            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[x - 1, y - 1] if y > 0 else max_neg_val

            value[x, y] = max(v_cur, v_prev) + value[x, y]

    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


@njit(parallel=True, cache=True, fastmath=True, nogil=True, boundscheck=False)
def maximum_path_c(
    paths: NDArray[np.int32],
    values: NDArray[np.float32],
    t_xs: NDArray[np.int32],
    t_ys: NDArray[np.int32],
    max_neg_val: float = -1e9,
) -> None:
    """Batch version using Numba parallel loops."""
    b = values.shape[0]
    for i in prange(b):
        maximum_path_each_numba(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
