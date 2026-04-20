import numpy as np
import torch
from nx.core import from_numpy
from utils import make_shape, torch_assert_array


class TestBackprop:
    def test_backprop_v1(self):
        print("backprop 1:")
        shape = make_shape()
        np1 = np.random.randn(*shape).astype(np.float32)
        np2 = np.random.randn(*shape).astype(np.float32)
        nx1 = from_numpy(np1)
        nx2 = from_numpy(np2)
        nx3 = nx1 + nx2
        nx3.grad_enabled = True
        nx4 = nx1 * nx2
        nx4.grad_enabled = True
        nx5 = nx3 + nx4
        nx5.grad_enabled = True
        nx6 = nx3 * nx4
        nx6.grad_enabled = True
        nx7 = nx5 + nx6
        nx7.grad_enabled = True
        nx8 = nx7.sum()
        nx8.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 + t2
        t3.retain_grad()
        t4 = t1 * t2
        t4.retain_grad()
        t5 = t3 + t4
        t5.retain_grad()
        t6 = t3 * t4
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)
        torch_assert_array(nx6.grad, t6.grad)
        torch_assert_array(nx7.grad, t7.grad)

    def test_backprop_v2(self):
        print("Testing complex unary(and one binary) operations chain:")
        shape = make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(
            np.float32
        )  # Positive values for log

        # Implementation: log(exp(x) * x) / x
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = nx1.exp()
        nx2.grad_enabled = True
        nx3 = nx2 * nx1
        nx3.grad_enabled = True
        nx4 = nx3.log()
        nx4.grad_enabled = True
        nx5 = nx4 / nx1
        nx5.grad_enabled = True
        nx6 = nx5.sum()
        nx6.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = t1.exp()
        t2.retain_grad()
        t3 = t2 * t1
        t3.retain_grad()
        t4 = t3.log()
        t4.retain_grad()
        t5 = t4 / t1
        t5.retain_grad()
        t6 = t5.sum()
        t6.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)

    def test_backprop_v3(self):
        print("Testing branched operations:")
        shape = make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = from_numpy(np2)
        nx2.grad_enabled = True
        nx3 = nx1.log()
        nx3.grad_enabled = True
        nx4 = nx2.exp()
        nx4.grad_enabled = True
        nx5 = nx3 * nx4
        nx5.grad_enabled = True
        nx6 = nx1 / nx2
        nx6.grad_enabled = True
        nx7 = nx5 + nx6
        nx7.grad_enabled = True
        nx8 = nx7.maximum(nx6)
        nx8.grad_enabled = True
        nx9 = nx8.sum()
        nx9.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1.log()
        t3.retain_grad()
        t4 = t2.exp()
        t4.retain_grad()
        t5 = t3 * t4
        t5.retain_grad()
        t6 = t1 / t2
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.maximum(t6)
        t8.retain_grad()
        t9 = t8.sum()
        t9.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)
        torch_assert_array(nx6.grad, t6.grad)
        torch_assert_array(nx7.grad, t7.grad)
        torch_assert_array(nx8.grad, t8.grad)

    def test_backprop_v4(self):
        print("Testing nested operations:")
        shape = make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation: log(exp(x1/x2) * recip(x1))
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = from_numpy(np2)
        nx2.grad_enabled = True
        nx3 = nx1 / nx2
        nx3.grad_enabled = True
        nx4 = nx3.exp()
        nx4.grad_enabled = True
        nx5 = nx1.recip()
        nx5.grad_enabled = True
        nx6 = nx4 * nx5
        nx6.grad_enabled = True
        nx7 = nx6.log()
        nx7.grad_enabled = True
        nx8 = nx7.sum()
        nx8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 / t2
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = 1.0 / t1
        t5.retain_grad()
        t6 = t4 * t5
        t6.retain_grad()
        t7 = t6.log()
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)
        torch_assert_array(nx6.grad, t6.grad)
        torch_assert_array(nx7.grad, t7.grad)

    def test_backprop_v5(self):
        print("Testing square and sqrt operations:")
        shape = make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation: sqrt(x^2) + x^2/sqrt(x)
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = nx1.sq()
        nx2.grad_enabled = True
        nx3 = nx2.sqrt()
        nx3.grad_enabled = True
        nx4 = nx1.sq()
        nx4.grad_enabled = True
        nx5 = nx1.sqrt()
        nx5.grad_enabled = True
        nx6 = nx4 / nx5
        nx6.grad_enabled = True
        nx7 = nx3 + nx6
        nx7.grad_enabled = True
        nx8 = nx7.sum()
        nx8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = t1 * t1
        t2.retain_grad()
        t3 = t2.sqrt()
        t3.retain_grad()
        t4 = t1 * t1
        t4.retain_grad()
        t5 = t1.sqrt()
        t5.retain_grad()
        t6 = t4 / t5
        t6.retain_grad()
        t7 = t3 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)
        torch_assert_array(nx6.grad, t6.grad)
        torch_assert_array(nx7.grad, t7.grad)

    def test_backprop_v8(self):
        print("Testing double backpropagation with complex operations:")
        shape = make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation
        # f(x1, x2) = log(sqrt(x1^2) * exp(x2/x1)) + (x1 * sqrt(x2))^2
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = from_numpy(np2)
        nx2.grad_enabled = True
        nx3 = nx1.sq()
        nx3.grad_enabled = True
        nx4 = nx3.sqrt()
        nx4.grad_enabled = True
        nx5 = nx2 / nx1
        nx5.grad_enabled = True
        nx6 = nx5.exp()
        nx6.grad_enabled = True
        nx7 = nx4 * nx6
        nx7.grad_enabled = True
        nx8 = nx7.minimum(nx4)
        nx8.grad_enabled = True
        nx9 = nx8.log()
        nx9.grad_enabled = True
        nx10 = nx2.sqrt()
        nx10.grad_enabled = True
        nx11 = nx1 * nx10
        nx11.grad_enabled = True
        nx12 = nx11.sq()
        nx12.grad_enabled = True
        nx13 = nx9 + nx12
        nx14 = nx13.sum()
        nx14.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 * t1
        t3.retain_grad()
        t4 = t3.sqrt()
        t4.retain_grad()
        t5 = t2 / t1
        t5.retain_grad()
        t6 = t5.exp()
        t6.retain_grad()
        t7 = t4 * t6
        t7.retain_grad()
        t8 = t7.minimum(t4)
        t8.retain_grad()
        t9 = t8.log()
        t9.retain_grad()
        t10 = t2.sqrt()
        t10.retain_grad()
        t11 = t1 * t10
        t11.retain_grad()
        t12 = t11.square()
        t12.retain_grad()
        t13 = t9 + t12
        t13.retain_grad()
        t14 = t13.sum()
        t14.backward(retain_graph=True)

        # Compare backward pass gradients
        print("\nChecking backward pass:")
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
        torch_assert_array(nx4.grad, t4.grad)
        torch_assert_array(nx5.grad, t5.grad)
        torch_assert_array(nx6.grad, t6.grad)
        torch_assert_array(nx7.grad, t7.grad)
        torch_assert_array(nx8.grad, t8.grad)
        torch_assert_array(nx9.grad, t9.grad)
        torch_assert_array(nx10.grad, t10.grad)
        torch_assert_array(nx11.grad, t11.grad)
        torch_assert_array(nx12.grad, t12.grad)

    def test_permute_binary_backprop(self):
        """Test backprop through permute and binary op"""
        # Forward: (2,3,4) -> (4,2,3) * (4,2,3)
        x = torch.randn(2, 3, 4, dtype=torch.float32)
        y = torch.randn(4, 2, 3, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = from_numpy(y.numpy())
        nx2.grad_enabled = True
        nx3 = nx1.permute([2, 0, 1]) * nx2
        nx4 = nx3.sum()
        nx4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1.permute(2, 0, 1) * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)

    def test_backprop_v6(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = nx1.permute([0, 2, 1, 3]).reshape([8, 3, 5])  # (2,4,3,5)  # (8,3,5)
        nx3 = nx2.exp()
        nx4 = nx3.sum()
        nx4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1.permute(0, 2, 1, 3).reshape(8, 3, 5)  # (2,4,3,5)  # (8,3,5)
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)

    def test_backprop_v7(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> matmul -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
        # TODO: can try doing matmul with broadcast
        y = torch.randn(8, 5, 2, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = from_numpy(y.numpy())
        nx2.grad_enabled = True
        nx3 = (
            nx1.permute([0, 2, 1, 3]).reshape([8, 3, 5]) @ nx2
        )  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        nx4 = nx3.exp()
        nx5 = nx4.sum()
        nx5.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = (
            t1.permute(0, 2, 1, 3).reshape(8, 3, 5) @ t2
        )  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)

    def test_slice_basic_backprop(self):
        """Test basic slicing backpropagation"""
        print("\nTesting basic slice backprop:")
        x = torch.randn(4, 6, 8, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = nx1[1:3, ::2, ::1]  # Basic slicing
        nx3 = nx2.sum()
        nx3.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:3, ::2, ::1]
        t2.retain_grad()
        t3 = t2.sum()
        t3.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)

    def test_slice_with_unary_backprop(self):
        """Test slicing combined with unary operations"""
        print("\nTesting slice with unary ops backprop:")
        x = torch.randn(3, 4, 5, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = nx1[::2, 1:3]  # Slice first
        nx3 = nx2.exp()  # Then unary op
        nx4 = nx3.sum()
        nx4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[::2, 1:3]
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)

    def test_slice_with_binary_backprop(self):
        """Test slicing combined with binary operations"""
        print("\nTesting slice with binary ops backprop:")

        x = torch.randn(4, 6, 8, dtype=torch.float32)
        y = torch.randn(2, 6, 8, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = from_numpy(y.numpy())
        nx2.grad_enabled = True
        nx3 = nx1[::2] * nx2  # Slice and multiply
        nx4 = nx3.sum()
        nx4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2] * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)

    def test_slice_chain_backprop(self):
        """Test chain of slice operations"""
        print("\nTesting slice chain backprop:")
        x = torch.randn(5, 6, 7, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = nx1[1:4, ::2]  # First slice
        nx3 = nx2[:, 1::2]  # Second slice
        nx4 = nx3.sum()
        nx4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:4, ::2]
        t2.retain_grad()
        t3 = t2[:, 1::2]
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)

    def test_slice_complex_chain_backprop(self):
        """Test complex chain with slicing, unary and binary operations"""
        print("\nTesting complex slice chain backprop:")

        x = torch.randn(4, 5, 6, dtype=torch.float32)
        y = torch.randn(2, 5, 3, dtype=torch.float32)

        # numx implementation
        nx1 = from_numpy(x.numpy())
        nx1.grad_enabled = True
        nx2 = from_numpy(y.numpy())
        nx2.grad_enabled = True
        nx3 = nx1[::2, :, ::2]  # Initial slice
        nx4 = nx3.exp()  # Unary op
        nx5 = nx4 * nx2  # Binary op
        nx6 = nx5[:, 1:4]  # Another slice
        nx7 = nx6.sum()
        nx7.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2, :, ::2]
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4 * t2
        t5.retain_grad()
        t6 = t5[:, 1:4]
        t6.retain_grad()
        t7 = t6.sum()
        t7.backward()

        # Compare gradients
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)

    def test_linear_backprop(self):
        """Test backprop through matmul"""
        print("\nTesting matmul backprop:")
        np1 = np.random.randn(64, 784).astype(np.float32)
        np2 = np.random.randn(10, 784).astype(np.float32)
        np3 = np.random.randn(10).astype(np.float32)
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        nx2 = from_numpy(np2)
        nx2.grad_enabled = True
        nx3 = from_numpy(np3)
        nx3.grad_enabled = True
        nx4 = nx1 @ nx2.transpose(-2, -1) + nx3
        nx5 = nx4.sum()
        nx5.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = torch.from_numpy(np3).requires_grad_(True)
        t4 = t1 @ t2.T + t3
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()
        torch_assert_array(nx4, t4)
        torch_assert_array(nx1.grad, t1.grad)
        torch_assert_array(nx2.grad, t2.grad)
        torch_assert_array(nx3.grad, t3.grad)
