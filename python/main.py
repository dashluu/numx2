import numpy as np
from nx.core import Array, b8, f32, from_numpy, full, i32, ones, zeros
from typing_extensions import Sequence


def run_basic(shape1: Sequence, shape2: Sequence):
    np1 = np.random.randn(*shape1).astype(np.float32)
    np2 = np.random.randn(*shape2).astype(np.float32)
    nx1 = from_numpy(np1)
    nx2 = from_numpy(np2)
    nx3 = nx1 + nx2
    np3 = np1 + np2
    assert tuple(nx3.view) == np3.shape
    assert np.allclose(nx3.numpy(), np3, atol=1e-3, rtol=0)


def run_gemm(shape1: Sequence, shape2: Sequence):
    np1 = np.random.randn(*shape1).astype(np.float32)
    np2 = np.random.randn(*shape2).astype(np.float32)
    nx1 = from_numpy(np1)
    nx2 = from_numpy(np2)
    nx3 = nx1 @ nx2
    np3 = np1 @ np2
    assert tuple(nx3.view) == np3.shape
    assert np.allclose(nx3.numpy(), np3, atol=1e-3, rtol=0)


if __name__ == "__main__":
