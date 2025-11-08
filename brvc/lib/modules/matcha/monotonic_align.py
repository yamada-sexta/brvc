import numpy as np
import torch
from numpy.typing import NDArray
from lib.modules.matcha.maximum_path import maximum_path_c


def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value: NDArray[np.float32] = value.data.cpu().numpy().astype(np.float32)
    path: NDArray[np.int32] = np.zeros_like(value).astype(np.int32)
    mask: NDArray[np.float32] = mask.data.cpu().numpy()

    t_x_max: NDArray[np.int32] = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max: NDArray[np.int32] = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def test(batch_size: int = 32, t_x: int = 64, t_y: int = 80) -> None:
    """
    Test correctness and performance of `maximum_path`.

    - Generates random test tensors
    - Verifies output shape and binary values
    - Benchmarks performance
    """
    import time

    print(f"\n[TEST] batch={batch_size}, t_x={t_x}, t_y={t_y}")

    # Generate random tensors
    value = torch.rand(batch_size, t_x, t_y, dtype=torch.float32)
    mask = (torch.rand(batch_size, t_x, t_y) > 0.2).float()

    # Warm-up (JIT compile if using Numba)
    _ = maximum_path(value, mask)

    # Measure runtime
    start = time.perf_counter()
    path = maximum_path(value, mask)
    elapsed = time.perf_counter() - start

    # Basic correctness checks
    assert path.shape == value.shape, "[Error] Output shape mismatch"
    assert torch.all((path == 0) | (path == 1)), "[Error] Path tensor is not binary"

    # Sanity check: each batch should have at least one '1' value
    non_empty = (path.sum(dim=(1, 2)) > 0).all().item()
    assert non_empty, "[Error] Empty path detected"

    print(f"[Info] Shape check: {tuple(path.shape)}")
    print(f"[Info] Binary check: PASS")
    print(f"[Info] Non-empty paths: PASS")
    print(f"[Info] Runtime: {elapsed * 1000:.2f} ms for batch={batch_size}")
    print(f"[Info] Speed per item: {elapsed / batch_size * 1000:.2f} ms/item")

if __name__ == "__main__":
    test()