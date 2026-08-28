"""Optimization module"""

from collections.abc import Sequence

import nx.core


class Optimizer:
    def __init__(self, lr: float) -> None:
        """Base optimizer"""

    def init_single(self, param: nx.core.Array, state: "std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, nx::core::Array, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const, nx::core::Array>>>") -> None:
        """Initialize state for a single parameter"""

    def apply_single(self, param: nx.core.Array, grad: nx.core.Array, state: "std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, nx::core::Array, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const, nx::core::Array>>>") -> None:
        """Update function for a single parameter"""

    def update(self, params: Sequence[nx.core.Array], states: Sequence["std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, nx::core::Array, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const, nx::core::Array>>>"]) -> None:
        """Update module parameters"""

class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False) -> None:
        """Stochastic Gradient Descent optimizer"""

class Adam(Optimizer):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08, bias_correction: bool = False) -> None:
        """Adam optimizer"""
