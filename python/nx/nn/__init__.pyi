import nx.core


def linear(x: nx.core.Array, weight: nx.core.Array) -> nx.core.Array:
    """Functional linear without bias"""

def linear_with_bias(x: nx.core.Array, weight: nx.core.Array, bias: nx.core.Array) -> nx.core.Array:
    """Functional linear with bias"""

def relu(x: nx.core.Array) -> nx.core.Array:
    """ReLU activation function"""

def onehot(x: nx.core.Array, num_classes: int = -1) -> nx.core.Array:
    """One-hot encode input array"""

def softmax(x: nx.core.Array, dim: int = -1) -> nx.core.Array:
    """Compute softmax for input array"""

def cross_entropy_loss(x: nx.core.Array, y: nx.core.Array) -> nx.core.Array:
    """Compute cross-entropy loss between input x and target y"""

class Module:
    def __init__(self) -> None:
        """Base module"""

    def add_parameter(self, param: nx.core.Array) -> None:
        """Add parameter to module"""

    def parameters(self) -> list[nx.core.Array]:
        """Get module parameters"""

    def forward(self, x: nx.core.Array) -> nx.core.Array:
        """Forward pass through module"""

    def __call__(self, x: nx.core.Array) -> nx.core.Array:
        """Forward pass through module"""

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Linear layer"""

    @property
    def weight(self) -> nx.core.Array:
        """Get linear layer weight"""

    @property
    def bias(self) -> nx.core.Array:
        """Get linear layer bias"""
