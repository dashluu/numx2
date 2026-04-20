import torch
from nx.core import Array, from_numpy
from utils import torch_assert_array


class TestReduce:
    def reduce_basic(self, op1, op2):
        # Create test data
        test_cases = [
            [2, 3, 4],
            [3, 4, 5],
            [31, 67, 18, 17],
            [1027, 64, 32],
            [5674, 3289],
            [1, 1],
        ]

        for shape in test_cases:
            t1 = torch.randn(shape, dtype=torch.float32)
            nx1 = from_numpy(t1.numpy())
            nx2: Array = op1(nx1)
            t2: torch.Tensor = op2(t1)
            torch_assert_array(nx2, t2)

    def reduce_2d(self, op1, op2):
        # Test cases with different dimensions
        shapes = [
            (2, 3),
            (19, 29),
            (37, 32),
            (47, 7),
            (57, 3),
            (297, 101),
            (5674, 3289),
            (256, 1),
            (1, 1),
            (1, 997),
        ]

        for shape in shapes:
            print(shape)
            for dim in range(2):
                t1 = torch.randn(*shape, dtype=torch.float32)
                nx1 = from_numpy(t1.numpy())
                nx2: Array = op1(nx1, [dim])
                t2: torch.Tensor = op2(t1, dim)
                torch_assert_array(nx2, t2)

    def reduce_multidim(self, op1, op2):
        # Test cases with different shapes
        shapes = [
            (2, 3, 6, 4),
            (19, 29, 5, 11, 7),
            (37, 32, 18, 9),
            (47, 7, 7, 1, 3),
            (57, 3, 12),
            (297, 101, 5, 6, 3),
            (5674, 3289, 1),
        ]
        dims = [[1, 2], [0, 1, 3], [2, 3], [1, 3, 4], [0, 1, 2], [0, 1, 4], [1]]

        for shape, dim in zip(shapes, dims):
            t1 = torch.randn(*shape, dtype=torch.float32)
            nx1 = from_numpy(t1.numpy())
            nx2: Array = op1(nx1, dim)
            t2: torch.Tensor = op2(t1, dim)
            torch_assert_array(nx2, t2)

    def arg_reduce_in_multidim_array(self, op1, op2):
        # Test cases with different shapes
        shapes = [
            (2, 3, 6, 4),
            (19, 29, 5, 11, 7),
            (37, 32, 18, 9),
            (47, 7, 7, 1, 3),
            (57, 3, 12),
            (297, 101, 5, 6, 3),
            (5674, 3289, 1),
        ]
        dims = [[3], [1], [2], [4], [0], [1], [0]]

        for shape, dim in zip(shapes, dims):
            t1 = torch.randn(*shape, dtype=torch.float32)
            nx1 = from_numpy(t1.numpy())
            nx2: Array = op1(nx1, dim)
            # Note: PyTorch does not currently support multiple-dimension argmax and argmin
            t2: torch.Tensor = op2(t1, dim=dim[0])
            torch_assert_array(nx2, t2)

    def test_sum_basic(self):
        """Test basic sum reduction without specified dimensions"""
        print("Testing basic sum reduction:")
        self.reduce_basic(lambda x: x.sum(), lambda x: x.sum().unsqueeze(dim=-1))

    def test_sum_2d(self):
        """Test sum reduction along the row and column dimension"""
        print("Testing sum reduction along the row and column dimension:")
        self.reduce_2d(
            lambda x, dim: x.sum(dim), lambda x, dim: x.sum(dim=dim).unsqueeze(dim=-1)
        )

    def test_sum_multidim(self):
        """Test sum reduction along multiple dimensions"""
        print("Testing sum reduction along multiple dimensions:")
        self.reduce_multidim(
            lambda x, dim: x.sum(dim), lambda x, dim: x.sum(dim=dim).unsqueeze(dim=-1)
        )

    def test_mean_basic(self):
        """Test basic mean reduction without specified dimensions"""
        print("Testing basic mean reduction:")
        self.reduce_basic(lambda x: x.mean(), lambda x: x.mean().unsqueeze(dim=-1))

    def test_mean_2d(self):
        """Test mean reduction along the row and column dimension"""
        print("Testing mean reduction along the row and column dimension:")
        self.reduce_2d(
            lambda x, dim: x.mean(dim), lambda x, dim: x.mean(dim=dim).unsqueeze(dim=-1)
        )

    def test_mean_multidim(self):
        """Test mean reduction along multiple dimensions"""
        print("Testing mean reduction along multiple dimensions:")
        self.reduce_multidim(
            lambda x, dim: x.mean(dim), lambda x, dim: x.mean(dim=dim).unsqueeze(dim=-1)
        )

    def test_max_basic(self):
        """Test basic max reduction without specified dimensions"""
        print("Testing basic max reduction:")
        self.reduce_basic(lambda x: x.max(), lambda x: x.max().unsqueeze(dim=-1))

    def test_max_2d(self):
        """Test max reduction along the column dimension"""
        print("Testing max reduction along the column dimension:")
        self.reduce_2d(
            lambda x, dim: x.max(dim), lambda x, dim: x.amax(dim=dim).unsqueeze(dim=-1)
        )

    def test_max_multidim(self):
        """Test max reduction along multiple dimensions"""
        print("Testing max reduction along multiple dimensions:")
        self.reduce_multidim(
            lambda x, dim: x.max(dim), lambda x, dim: x.amax(dim=dim).unsqueeze(dim=-1)
        )

    def test_min_basic(self):
        """Test basic min reduction without specified dimensions"""
        print("Testing basic min reduction:")
        self.reduce_basic(lambda x: x.min(), lambda x: x.min().unsqueeze(dim=-1))

    def test_min_2d(self):
        """Test min reduction along the column dimension"""
        print("Testing min reduction along the column dimension:")
        self.reduce_2d(
            lambda x, dim: x.min(dim), lambda x, dim: x.amin(dim=dim).unsqueeze(dim=-1)
        )

    def test_min_multidim(self):
        """Test min reduction along multiple dimensions"""
        print("Testing min reduction along multiple dimensions:")
        self.reduce_multidim(
            lambda x, dim: x.min(dim), lambda x, dim: x.amin(dim=dim).unsqueeze(dim=-1)
        )

    def test_argmax_basic(self):
        """Test basic argmax reduction without specified dimensions"""
        print("Testing basic argmax reduction:")
        self.reduce_basic(
            lambda x: x.argmax(),
            lambda x: x.argmax().type(torch.int32).unsqueeze(dim=-1),
        )

    def test_argmax_2d(self):
        """Test argmax reduction along the column dimension"""
        print("Testing argmax reduction along the column dimension:")
        self.reduce_2d(
            lambda x, dim: x.argmax(dim),
            lambda x, dim: x.argmax(dim=dim).type(torch.int32).unsqueeze(dim=-1),
        )

    def test_argmax_multidim(self):
        """Test argmax reduction along multiple dimensions"""
        print("Testing argmax reduction along multiple dimensions:")
        self.arg_reduce_in_multidim_array(
            lambda x, dim: x.argmax(dim),
            lambda x, dim: x.argmax(dim=dim).type(torch.int32).unsqueeze(dim=-1),
        )

    def test_argmin_basic(self):
        """Test basic argmin reduction without specified dimensions"""
        print("Testing basic argmin reduction:")
        self.reduce_basic(
            lambda x: x.argmin(),
            lambda x: x.argmin().type(torch.int32).unsqueeze(dim=-1),
        )

    def test_argmin_2d(self):
        """Test argmin reduction along the column dimension"""
        print("Testing argmin reduction along the column dimension:")
        self.reduce_2d(
            lambda x, dim: x.argmin(dim),
            lambda x, dim: x.argmin(dim=dim).type(torch.int32).unsqueeze(dim=-1),
        )

    def test_argmin_multidim(self):
        """Test argmin reduction along multiple dimensions"""
        print("Testing argmin reduction along multiple dimensions:")
        self.arg_reduce_in_multidim_array(
            lambda x, dim: x.argmin(dim),
            lambda x, dim: x.argmin(dim=dim).type(torch.int32).unsqueeze(dim=-1),
        )
