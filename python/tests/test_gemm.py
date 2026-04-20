import numpy as np
from nx.core import from_numpy
from utils import np_assert_array, np_assert_scalar


class TestGemm:
    def test_gevv(self):
        """Test gevv function"""
        print("gevv:")
        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([12], [12]),
            ([1], [1]),
            ([1024], [1024]),
            ([3196], [3196]),
            ([197], [197]),
            ([528], [528]),
            ([43], [43]),
        ]
        for shape1, shape2 in test_cases:
            print(f"Testing shapes: {shape1} @ {shape2}")
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3 = nx1 @ nx2
            np3 = np1 @ np2
            np_assert_scalar(nx3, np3)

    def test_gemm2d(self):
        """Test matrix multiplication for 2D arrays"""
        print("gemm 2d:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([2, 3], [3, 4]),  # Basic matrix multiplication
            ([1, 4], [4, 5]),  # Single row matrix
            ([3, 2], [2, 1]),  # Result is a column matrix
            ([5, 5], [5, 5]),  # Square matrices
            ([1, 1], [1, 1]),  # 1x1 matrices,
            # More common cases
            ([67, 99], [99, 35]),
            ([1024, 519], [519, 267]),
            ([31, 27], [27, 75]),
            ([51, 7], [7, 29]),
            ([128, 64], [64, 256]),
            ([256, 256], [256, 256]),
        ]

        for shape1, shape2 in test_cases:
            print(f"Testing shapes: {shape1} @ {shape2}")
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3 = nx1 @ nx2
            np3 = np1 @ np2
            np_assert_array(nx3, np3)

    def test_gemm3d(self):
        """Test matrix multiplication for 3D arrays (batched gemm)"""
        print("Testing 3D matrix multiplication:")

        # Test cases: [(shape1, shape2, description)]
        test_cases = [
            # Basic batch gemm
            ([4, 2, 3], [4, 3, 4], "Standard batch size"),
            ([1, 2, 3], [1, 3, 4], "Single batch"),
            ([10, 3, 3], [10, 3, 3], "Square matrices batch"),
            # Broadcasting cases
            ([1, 2, 3], [5, 3, 4], "Broadcast first dim"),
            ([5, 2, 3], [1, 3, 4], "Broadcast second dim"),
            ([7, 1, 3], [7, 3, 5], "Batch with singular dimension"),
            # Edge cases
            ([3, 1, 4], [3, 4, 1], "Result has singular dimension"),
            ([2, 5, 1], [2, 1, 3], "Inner dimension is 1"),
            ([1, 1, 1], [1, 1, 1], "All dimensions are 1"),
            # More common cases
            ([27, 68, 25], [27, 25, 19], "Common case 1"),
            ([11, 32, 49], [11, 49, 58], "Common case 2"),
            ([51, 29, 7], [51, 7, 30], "Common case 3"),
        ]

        for shape1, shape2, desc in test_cases:
            print(f"Testing {desc}:")
            print(f"Shapes: {shape1} @ {shape2}")
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3 = nx1 @ nx2
            np3 = np1 @ np2
            np_assert_array(nx3, np3)

    def test_multidim_gemm(self):
        """Test multi-dimensional matrix multiplication"""
        print("Testing multi-dimensional matrix multiplication:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([1, 5, 4, 2, 3], [5, 1, 4, 3, 4]),
            ([1, 1, 2, 3], [5, 2, 3, 4]),
            ([1, 3, 7, 3, 17], [10, 3, 1, 17, 6]),
            ([13, 4, 2, 9, 1], [1, 1, 2, 1, 8]),
        ]

        for shape1, shape2 in test_cases:
            print(f"Shapes: {shape1} @ {shape2}")
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            nx1 = from_numpy(np1)
            nx2 = from_numpy(np2)
            nx3 = nx1 @ nx2
            np3 = np1 @ np2
            np_assert_array(nx3, np3)
