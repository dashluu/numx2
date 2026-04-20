from __future__ import annotations

import operator

import numpy as np
from nx.core import Array, from_numpy
from utils import np_assert_array


class TestBinary:
    def binary_no_broadcast(self, name: str, op1, op2):
        print(f"{name}:")

        test_cases = [
            [48, 23, 14, 28],
            [1, 37, 17, 4, 29],
            [6, 54, 13, 89],
            [28, 32, 64],
            [25, 32, 1, 9],
            [1, 2, 1, 1, 15, 10],
        ]

        for shape in test_cases:
            print(f"Testing shape: {shape}")
            np1 = np.random.randn(*shape).astype(np.float32)
            np2 = np.random.randn(*shape).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3: Array = op1(nx1, nx2)
            np3: np.ndarray = op2(np1, np2)
            np_assert_array(nx3, np3)

    def binary_with_broadcast(self, name: str, op1, op2):
        print(f"{name} with broadcast:")

        # Test cases with different broadcasting scenarios
        test_cases = [
            # [shape1, shape2, result_shape]
            ([2, 1, 4], [3, 4], [2, 3, 4]),  # Left broadcast
            ([1, 5], [2, 1, 5], [2, 1, 5]),  # Right broadcast
            ([3, 1, 1], [1, 4, 5], [3, 4, 5]),  # Both broadcast
            ([1], [2, 3, 4], [2, 3, 4]),  # Scalar to array
            ([2, 3, 4], [1], [2, 3, 4]),  # Array to scalar
            ([3, 1, 19, 1, 1], [1, 47, 19, 63, 1], [3, 47, 19, 63, 1]),
            ([1, 2], [1, 47, 19, 63, 1], [1, 47, 19, 63, 2]),
        ]

        for shape1, shape2, expected_shape in test_cases:
            print(f"Testing shapes: {shape1}, {shape2} -> {expected_shape}")
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3: Array = op1(nx1, nx2)
            np3: np.ndarray = op2(np1, np2)
            np_assert_array(nx3, np3)

    def binary_inplace(self, name: str, op1, op2):
        print(f"{name} inplace:")

        test_cases = [
            [48, 23, 14, 28],
            [1, 37, 17, 4, 29],
            [6, 54, 13, 89],
            [28, 32, 64],
            [25, 32, 1, 9],
            [1, 2, 1, 1, 15, 10],
        ]

        for shape in test_cases:
            print(f"Testing shape: {shape}")
            # Generate inputs
            np1 = np.random.randn(*shape).astype(np.float32)
            np2 = np.random.randn(*shape).astype(np.float32)

            # Create arrays
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)

            # Apply inplace operation
            nx3: Array = op1(nx1, nx2)  # nx1 += nx2, etc.
            nx3 = op1(nx3, nx2)  # Second time to make sure it is updated.

            # Compare with NumPy
            np3: np.ndarray = op2(np1.copy(), np2)  # np1_copy += np2, etc.
            np3 = op2(np3, np2)  # Second time
            np_assert_array(nx3, np3)

    def binary_inplace_broadcast(self, name: str, op1, op2):
        print(f"{name} inplace broadcast:")

        test_cases = [
            # [lshape, rshape] -> result shape will be lshape
            ([2, 3, 4], [4]),  # Broadcast scalar to 3D
            ([3, 4, 5], [1, 5]),  # Broadcast from 2D to 3D
            ([2, 4, 6], [4, 1]),  # Broadcast with ones
            ([5, 5, 5], [1, 5, 1]),  # Broadcast with ones in multiple dims
            ([4, 3, 2], [3, 1]),  # Partial broadcast with ones
            ([3, 47, 19, 63, 1], [1, 1, 19, 63, 1]),
            ([1, 47, 19, 63, 2], [1, 2]),
        ]

        for shape1, shape2 in test_cases:
            print(f"Testing: {shape1} @= {shape2}")

            # Generate inputs
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)

            # Create arrays
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)

            # Apply inplace operation
            nx3: Array = op1(nx1, nx2)
            # Compare with NumPy
            np3: np.ndarray = op2(np1.copy(), np2)
            np_assert_array(nx3, np3)

    def test_add(self):
        self.binary_no_broadcast("add", operator.add, operator.add)

    def test_sub(self):
        self.binary_no_broadcast("sub", operator.sub, operator.sub)

    def test_mul(self):
        self.binary_no_broadcast("mul", operator.mul, operator.mul)

    def test_div(self):
        self.binary_no_broadcast("div", operator.truediv, operator.truediv)

    def test_minimum(self):
        self.binary_no_broadcast(
            "minimum", lambda x, y: x.minimum(y), lambda x, y: np.minimum(x, y)
        )

    def test_maximum(self):
        self.binary_no_broadcast(
            "maximum", lambda x, y: x.maximum(y), lambda x, y: np.maximum(x, y)
        )

    def test_add_broadcast(self):
        self.binary_with_broadcast("add", operator.add, operator.add)

    def test_sub_broadcast(self):
        self.binary_with_broadcast("sub", operator.sub, operator.sub)

    def test_mul_broadcast(self):
        self.binary_with_broadcast("mul", operator.mul, operator.mul)

    def test_div_broadcast(self):
        self.binary_with_broadcast("div", operator.truediv, operator.truediv)

    def test_add_inplace(self):
        self.binary_inplace("add", operator.iadd, operator.iadd)

    def test_sub_inplace(self):
        self.binary_inplace("sub", operator.isub, operator.isub)

    def test_mul_inplace(self):
        self.binary_inplace("mul", operator.imul, operator.imul)

    def test_div_inplace(self):
        self.binary_inplace("div", operator.itruediv, operator.itruediv)

    def test_add_inplace_broadcast(self):
        self.binary_inplace_broadcast("add", operator.iadd, operator.iadd)

    def test_sub_inplace_broadcast(self):
        self.binary_inplace_broadcast("sub", operator.isub, operator.isub)

    def test_mul_inplace_broadcast(self):
        self.binary_inplace_broadcast("mul", operator.imul, operator.imul)

    def test_div_inplace_broadcast(self):
        self.binary_inplace_broadcast("div", operator.itruediv, operator.itruediv)
