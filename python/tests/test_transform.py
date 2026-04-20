import numpy as np
from nx.core import from_numpy
from utils import np_assert_array


class TestTransform:
    def test_slice_v1(self):
        print("slice 1:")
        shape = [np.random.randint(1, 50) for _ in range(3)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1[::, ::, ::]
        np2 = np1[::, ::, ::]
        np_assert_array(nx2, np2)

    def test_slice_v2(self):
        print("slice 2:")
        shape = [np.random.randint(4, 50) for _ in range(4)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1[1::4, :3:2, 2::3]
        np2 = np1[1::4, :3:2, 2::3]
        np_assert_array(nx2, np2)

    def test_slice_v3(self):
        print("slice 3:")
        shape = [np.random.randint(4, 50) for _ in range(4)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1[1::, ::2, 3:0:-2]
        np2 = np1[1::, ::2, 3:0:-2]
        np_assert_array(nx2, np2)

    def test_slice_v4(self):
        print("slice 4:")
        shape = [np.random.randint(10, 50) for _ in range(4)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1[1:0:-4, 9:3:-2, 2::3]
        np2 = np1[1:0:-4, 9:3:-2, 2::3]
        np_assert_array(nx2, np2)

    def test_transpose_start(self):
        print("transpose at the start:")
        shape = [np.random.randint(3, 10) for _ in range(4)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1.transpose(0, 2)
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[0 : 2 + 1] = order[0 : 2 + 1][::-1]  # [2,1,0,3]
        np2 = np.transpose(np1, order)
        np_assert_array(nx2, np2)

    def test_transpose_mid(self):
        print("transpose in the middle:")
        shape = [np.random.randint(3, 10) for _ in range(6)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1.transpose(1, -2)
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[1:-1] = order[1:-1][::-1]  # [0,3,2,1]
        np2 = np.transpose(np1, order)
        np_assert_array(nx2, np2)

    def test_transpose_end(self):
        print("transpose at the end:")
        shape = [np.random.randint(3, 10) for _ in range(5)]
        np1 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = nx1.transpose(-3, -1)
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[-3:] = order[-3:][::-1]  # [0,3,2,1]
        np2 = np.transpose(np1, order)
        np_assert_array(nx2, np2)

    def test_permute(self):
        print("Testing permute operations:")

        # Test cases: [(shape, permutation)]
        test_cases = [
            # Basic permutations
            ([2, 3, 4], [2, 0, 1]),  # 3D rotation
            ([2, 3, 4, 5], [3, 2, 1, 0]),  # Complete reverse
            ([2, 3, 4, 5], [0, 2, 1, 3]),  # Middle swap
            # Edge cases
            ([1, 2, 3], [2, 1, 0]),  # With dimension size 1
            ([5, 1, 1, 4], [0, 2, 1, 3]),  # Multiple size-1 dimensions
            ([2, 3], [1, 0]),  # 2D transpose
        ]

        for shape, permutation in test_cases:
            print(f"Testing shape {shape} with permutation {permutation}")

            # Create test data
            np1 = np.random.randn(*shape).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = nx1.permute(permutation)
            np2 = np.transpose(np1, permutation)
            np_assert_array(nx2, np2)

    def test_flatten(self):
        print("Testing flatten operations:")

        # Test cases: [(shape, start_dim, end_dim, expected_shape)]
        test_cases = [
            # Basic flattening
            ([2, 3, 4], 0, -1, [24]),  # Flatten all
            ([2, 3, 4, 5], 1, 2, [2, 12, 5]),  # Middle flatten
            ([2, 3, 4, 5], 0, 1, [6, 4, 5]),  # Start flatten
            ([2, 3, 4, 5], -2, -1, [2, 3, 20]),  # End flatten
            # Edge cases
            ([1, 2, 3, 4], 1, 3, [1, 24]),  # With leading 1
            ([2, 1, 3, 1], 1, 2, [2, 3, 1]),  # With middle 1s
            ([5], 0, 0, [5]),  # Single dimension
        ]

        for shape, start, end, expected in test_cases:
            print(f"Testing shape {shape} flatten({start},{end})")

            # Create test data
            np1 = np.random.randn(*shape).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = nx1.flatten(start, end)
            np2 = np1.reshape(expected)
            np_assert_array(nx2, np2)
