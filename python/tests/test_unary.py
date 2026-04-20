from __future__ import annotations

import numpy as np
from nx.core import Array, from_numpy
from utils import nonzero_randn, np_assert_array, positive_randn, randn


class TestUnary:
    def unary_no_broadcast(self, name: str, op1, op2, rand_fn=randn):
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
            np1 = rand_fn(shape)
            nx1 = from_numpy(np1)
            nx2: Array = op1(nx1)
            np2: np.ndarray = op2(np1)
            np_assert_array(nx2, np2)

    def unary_with_slicing(self, name: str, op1, op2, rand_fn=randn):
        print(f"{name} with slicing:")

        # Test cases with different slicing patterns
        test_cases = [
            # [shape, slices] -> creates non-contiguous tensors
            ([4, 4], (slice(None, None, 2), slice(None))),  # Skip every other row
            ([4, 6], (slice(None), slice(None, None, 2))),  # Skip every other column
            ([4, 4, 4], (slice(None), slice(1, 3), slice(None))),  # Middle slice
            ([6, 6], (slice(None, None, 3), slice(1, None, 2))),  # Complex slicing
        ]

        for shape, slices in test_cases:
            print(f"Testing shape: {shape}, slices: {slices}")
            np1 = rand_fn(shape)
            nx1 = from_numpy(np1)
            # Create non-contiguous array using slicing
            nx2 = nx1[slices]
            nx3: Array = op1(nx2)  # Apply unary operation
            # Compare with NumPy
            np2 = np1[slices]  # Apply same slicing
            np3: np.ndarray = op2(np2)  # Apply same operation
            np_assert_array(nx3, np3)

    def unary_inplace(self, name: str, op1, op2, rand_fn=randn):
        print(f"{name} inplace:")

        # Test different shapes
        test_cases = [
            [5],  # 1D
            [2, 3],  # 2D
            [2, 3, 4],  # 3D
            [1, 2, 3, 4],  # 4D with leading 1
            [5, 1, 4],  # 3D with middle 1
        ]

        for shape in test_cases:
            print(f"Testing shape: {shape}")
            # Generate input
            np1 = rand_fn(shape)
            # Create array
            nx1 = from_numpy(np1)
            # Apply inplace operation
            nx2: Array = op1(nx1)
            # Compare with NumPy
            np2 = op2(np1.copy())
            np_assert_array(nx2, np2)

    def test_exp(self):
        self.unary_no_broadcast("exp", Array.exp, np.exp)

    def test_neg(self):
        self.unary_no_broadcast("neg", Array.neg, np.negative)

    def test_log(self):
        self.unary_no_broadcast("log", Array.log, np.log, rand_fn=positive_randn)

    def test_recip(self):
        self.unary_no_broadcast(
            "recip", Array.recip, np.reciprocal, rand_fn=nonzero_randn
        )

    def test_exp_with_slicing(self):
        self.unary_with_slicing("exp", Array.exp, np.exp)

    def test_neg_with_slicing(self):
        self.unary_with_slicing("neg", Array.neg, np.negative)

    def test_log_with_slicing(self):
        self.unary_with_slicing("log", Array.log, np.log, rand_fn=positive_randn)

    def test_exp_inplace(self):
        def exp_inplace(x: Array):
            return x.exp(in_place=True)

        self.unary_inplace("exp", exp_inplace, np.exp)

    def test_sqrt_inplace(self):
        def sqrt_inplace(x: Array):
            return x.sqrt(in_place=True)

        self.unary_inplace("sqrt", sqrt_inplace, np.sqrt, rand_fn=positive_randn)

    def test_neg_inplace(self):
        def neg_inplace(x: Array):
            return x.neg(in_place=True)

        self.unary_inplace("neg", neg_inplace, np.negative)

    def test_recip_inplace(self):
        def recip_inplace(x: Array):
            return x.recip(in_place=True)

        self.unary_inplace("recip", recip_inplace, np.reciprocal, rand_fn=nonzero_randn)

    def test_log_inplace(self):
        def log_inplace(x: Array):
            return x.log(in_place=True)

        self.unary_inplace("log", log_inplace, np.log, rand_fn=positive_randn)
