from __future__ import annotations

import numpy as np
import nx.nn as nn
import nx.optim as optim
import torch
from mnist import MnistModel
from nx.core import from_numpy
from utils import torch_assert_array, torch_assert_scalar


class TestBasicNN:
    def test_linear(self):
        np1 = np.random.randn(64, 784).astype(np.float32)
        nx1 = from_numpy(np1)
        t1 = torch.from_numpy(np1)
        linear = nn.Linear(784, 10)
        nx_weight = linear.weight
        nx_bias = linear.bias
        torch_weight: torch.Tensor = nx_weight.torch()
        torch_weight.requires_grad_(True)
        torch_weight.retain_grad()
        torch_bias: torch.Tensor = nx_bias.torch()
        torch_bias.requires_grad_(True)
        torch_bias.retain_grad()
        nx2 = linear(nx1).sum()
        t2 = (t1 @ torch_weight.T + torch_bias).sum()
        nx2.backward()
        t2.backward()
        torch_assert_scalar(nx2, t2)

    def test_relu(self):
        np1 = np.random.randn(64, 10).astype(np.float32)
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        t1 = torch.from_numpy(np1)
        t1.requires_grad_(True)
        nx2 = nn.relu(nx1)
        nx3 = nx2.sum()
        t2 = torch.relu(t1)
        t3 = t2.sum()
        nx3.backward()
        t3.backward()
        torch_assert_array(nx2, t2)
        torch_assert_array(nx1.grad, t1.grad)

    def test_onehot(self):
        np1 = np.random.randint(0, 10, (64,), dtype=np.int32)
        nx1 = from_numpy(np1)
        t1 = torch.from_numpy(np1).type(torch.int64)
        nx2 = nn.onehot(nx1)
        t2 = torch.nn.functional.one_hot(t1, num_classes=10).type(torch.int32)
        torch_assert_array(nx2, t2)

    def test_softmax(self):
        np1 = np.random.randn(64, 10).astype(np.float32)
        nx1 = from_numpy(np1)
        nx1.grad_enabled = True
        t1 = torch.from_numpy(np1)
        t1.requires_grad_(True)
        nx2 = nn.softmax(nx1)
        nx3 = nx2.sum()
        t2 = torch.softmax(t1, dim=-1)
        t3 = t2.sum()
        nx3.backward()
        t3.backward()
        torch_assert_array(nx2, t2)
        torch_assert_array(nx1.grad, t1.grad)

    def test_cross_entropy(self):
        np_input = np.random.randn(64, 10).astype(np.float32)
        np_label = np.random.randint(0, 10, (64,), dtype=np.int32)
        nx_input = from_numpy(np_input)
        nx_input.grad_enabled = True
        torch_input = torch.from_numpy(np_input)
        torch_input.requires_grad_(True)
        nx_label = from_numpy(np_label)
        torch_label = torch.from_numpy(np_label).type(torch.int64)
        nx_loss = nn.cross_entropy_loss(nx_input, nx_label)
        torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_input, torch_label)
        nx_loss.backward()
        torch_loss.backward()
        torch_assert_scalar(nx_loss, torch_loss)
        torch_assert_array(nx_input.grad, torch_input.grad)

    def test_single_pass(self):
        # Input data
        np_input = np.random.randn(64, 784).astype(np.float32)
        np_label = np.random.randint(0, 10, (64,), dtype=np.int32)

        # Array implementation
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        nx_model = MnistModel()
        nx_model.linear1.weight.grad_enabled = True
        nx_model.linear1.bias.grad_enabled = True
        nx_model.linear2.weight.grad_enabled = True
        nx_model.linear2.bias.grad_enabled = True

        # PyTorch implementation
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

        # Share weights between Array and PyTorch
        # First layer
        w1: torch.Tensor = nx_model.linear1.weight.torch()
        b1: torch.Tensor = nx_model.linear1.bias.torch()
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        # Second layer
        w2: torch.Tensor = nx_model.linear2.weight.torch()
        b2: torch.Tensor = nx_model.linear2.bias.torch()
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)

        # Forward pass
        torch_input = torch.from_numpy(np_input)
        torch_label = torch.from_numpy(np_label).type(torch.int64)

        nx_logits = nx_model(nx_input)
        torch_logits = torch_model(torch_input)

        # Loss computation
        nx_loss = nn.cross_entropy_loss(nx_logits, nx_label)
        torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(
            torch_logits, torch_label
        )

        # Backward pass
        nx_loss.backward()
        torch_loss.backward()

        # Compare results
        torch_assert_array(nx_logits, torch_logits)
        torch_assert_scalar(nx_loss, torch_loss)

        # Compare gradients
        torch_assert_array(nx_model.linear1.weight.grad, torch_model[0].weight.grad)
        torch_assert_array(nx_model.linear1.bias.grad, torch_model[0].bias.grad)
        torch_assert_array(nx_model.linear2.weight.grad, torch_model[2].weight.grad)
        torch_assert_array(nx_model.linear2.bias.grad, torch_model[2].bias.grad)

    def test_multipass_with_optimizer(self):
        nx_model = MnistModel()
        nx_model.linear1.weight.grad_enabled = True
        nx_model.linear1.bias.grad_enabled = True
        nx_model.linear2.weight.grad_enabled = True
        nx_model.linear2.bias.grad_enabled = True
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

        w1: torch.Tensor = nx_model.linear1.weight.torch()
        b1: torch.Tensor = nx_model.linear1.bias.torch()
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        w2: torch.Tensor = nx_model.linear2.weight.torch()
        b2: torch.Tensor = nx_model.linear2.bias.torch()
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)
        nx_loss_fn = nn.cross_entropy_loss
        torch_loss_fn = torch.nn.CrossEntropyLoss()
        nx_optimizer = optim.SGD(lr=1)
        torch_optimizer = torch.optim.SGD(
            torch_model.parameters(), lr=1, momentum=0, weight_decay=0
        )

        for _ in range(4):
            # Input data
            np_input = np.random.randn(64, 784).astype(np.float32)
            np_label = np.random.randint(0, 10, (64,), dtype=np.int32)

            # Array implementation
            nx_input = from_numpy(np_input)
            nx_label = from_numpy(np_label)
            # PyTorch implementation
            torch_input = torch.from_numpy(np_input)
            torch_label = torch.from_numpy(np_label).type(torch.int64)

            # Loss computation
            nx_logits = nx_model(nx_input)
            nx_loss = nx_loss_fn(nx_logits, nx_label)
            torch_logits = torch_model(torch_input)
            torch_loss: torch.Tensor = torch_loss_fn(torch_logits, torch_label)
            # print(nx_loss.item(), torch_loss.item())

            # Backward pass and parameters update
            nx_loss.backward()
            torch_optimizer.zero_grad()
            torch_loss.backward()
            # Compare losses
            torch_assert_scalar(nx_loss, torch_loss)

            # print(nx_model.linear1.weight.torch())
            # print(torch_model[0].weight)
            # print(nx_model.linear1.weight.grad.torch())
            # print(torch_model[0].weight.grad)

            # Update parameters
            nx_optimizer.update(nx_model.parameters(), [])
            torch_optimizer.step()

            # print(nx_model.linear1.weight.torch())
            # print(torch_model[0].weight)

            # Compare updated weights and biases
            torch_assert_array(nx_model.linear1.weight.grad, torch_model[0].weight.grad)
            torch_assert_array(nx_model.linear1.bias.grad, torch_model[0].bias.grad)
            torch_assert_array(nx_model.linear2.weight.grad, torch_model[2].weight.grad)
            torch_assert_array(nx_model.linear2.bias.grad, torch_model[2].bias.grad)
            torch_assert_array(nx_model.linear1.weight, torch_model[0].weight)
            torch_assert_array(nx_model.linear1.bias, torch_model[0].bias)
            torch_assert_array(nx_model.linear2.weight, torch_model[2].weight)
            torch_assert_array(nx_model.linear2.bias, torch_model[2].bias)
