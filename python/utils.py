import numpy as np
import torch
from nx.core import Array


class ArrayGenerator:
    def generate(self, shape: list[int], dtype=np.float32) -> np.array:
        raise NotImplementedError


class RandnGenerator(ArrayGenerator):
    def generate(self, shape: list[int], dtype=np.float32) -> np.array:
        return np.random.randn(*shape).astype(dtype)


class NonzeroRandn(ArrayGenerator):
    def generate(self, shape: list[int], dtype=np.float32) -> np.array:
        array = np.random.randn(*shape).astype(dtype)
        # Replace zeros with small random values
        zero_mask = array == 0
        array[zero_mask] = np.random.uniform(0.1, 1.0, size=np.count_nonzero(zero_mask))
        return array


class PositiveRandn(NonzeroRandn):
    def generate(self, shape: list[int], dtype=np.float32) -> np.array:
        return np.abs(super().generate(shape, dtype))


class UniformGenerator(ArrayGenerator):
    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def generate(self, shape: list[int], dtype=np.float32) -> np.array:
        return np.random.uniform(low=self.low, high=self.high, size=shape).astype(dtype)


def make_shape(dim_low=1, dim_high=5, shape_low=1, shape_high=100) -> list[int]:
    n = np.random.randint(dim_low, dim_high)
    shape = [np.random.randint(shape_low, shape_high) for _ in range(n)]
    return shape


def np_assert_scalar(array: Array, ndarray: np.ndarray, atol=1e-4, rtol=1e-4):
    assert np.allclose(array.item(), ndarray.item(), atol=atol, rtol=rtol)


def np_assert_array(array: Array, ndarray: np.ndarray, atol=1e-4, rtol=1e-4):
    assert np.allclose(array.numpy(), ndarray, atol=atol, rtol=rtol)


def torch_assert_scalar(array: Array, tensor: torch.Tensor, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(array.item(), tensor.item(), atol=atol, rtol=rtol)


def torch_assert_array(array: Array, tensor: torch.Tensor, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(array.torch(), tensor, atol=atol, rtol=rtol)
