"""Core module"""

from collections.abc import Sequence
import enum

from numpy.typing import NDArray


class DTypeKind(enum.Enum):
    BOOL = 8

    SIGNED = 2

    UNSIGNED = 4

    FLOAT = 1

class DType:
    @property
    def name(self) -> str:
        """Get data type's name as string"""

    @property
    def size(self) -> int:
        """Get data type's size in bytes"""

    @property
    def category(self) -> DTypeKind:
        """Get data type's category"""

class F32(DType):
    """32-bit floating point dtype"""

class I32(DType):
    """32-bit integer dtype"""

class B8(DType):
    """Boolean dtype"""

f32: F32 = ...

i32: I32 = ...

b8: B8 = ...

class Shape:
    @property
    def offset(self) -> int:
        """Get shape's offset"""

    @property
    def view(self) -> list[int]:
        """Get shape's view"""

    @property
    def stride(self) -> list[int]:
        """Get shape's stride"""

    @property
    def ndim(self) -> int:
        """Get shape's number of dimensions"""

    @property
    def numel(self) -> int:
        """Get shape's total number of elements"""

    def __str__(self) -> str:
        """String representation of shape"""

class DeviceKind(enum.Enum):
    CPU = 0

    MPS = 1

class Device:
    @property
    def kind(self) -> DeviceKind:
        """Get device's kind"""

    @property
    def id(self) -> int:
        """Get device's ID"""

    def __str__(self) -> str:
        """String representation of device"""

class Array:
    @property
    def shape(self) -> Shape:
        """Get array's shape"""

    @property
    def dtype(self) -> DType:
        """Get array's data type"""

    @property
    def device(self) -> Device:
        """Get array's device"""

    @property
    def grad(self) -> Array | None:
        """Get array's gradient"""

    @property
    def ndim(self) -> int:
        """Get array's number of dimensions"""

    @property
    def numel(self) -> int:
        """Get array's total number of elements"""

    @property
    def offset(self) -> int:
        """Get array's offset"""

    @property
    def view(self) -> list[int]:
        """Get array's view"""

    @property
    def stride(self) -> list[int]:
        """Get array's stride"""

    @property
    def ptr(self) -> int:
        """Get array's raw data pointer"""

    @property
    def itemsize(self) -> int:
        """Get array's element size in bytes"""

    @property
    def nbytes(self) -> int:
        """Get array's total size in bytes"""

    def size(self, dim: int) -> int:
        """Get array's number of elements in certain dimension"""

    def __len__(self) -> int:
        """
        Get array's length, that is, the number of elements in the first dimension
        """

    @property
    def is_parameter(self) -> bool:
        """Check if array is a parameter"""

    @property
    def is_contiguous(self) -> bool:
        """Check if array is contiguous"""

    @property
    def grad_enabled(self) -> bool:
        """
        Get/set array's gradient tracking, setter can only be used before compilation and forwarding, otherwise, there is no effect
        """

    @grad_enabled.setter
    def grad_enabled(self, enabled: bool) -> None: ...

    def numpy(self) -> NDArray:
        """Convert array to NumPy array"""

    def torch(self) -> NDArray:
        """Convert array to PyTorch tensor"""

    def item(self) -> object:
        """Get array's only value"""

    def __add__(self, rhs: object) -> Array:
        """Add two arrays element-wise"""

    def __radd__(self, rhs: object) -> Array:
        """Add two arrays element-wise"""

    def __sub__(self, rhs: object) -> Array:
        """Subtract two arrays element-wise"""

    def __rsub__(self, rhs: object) -> Array:
        """Subtract two arrays element-wise"""

    def __mul__(self, rhs: object) -> Array:
        """Multiply two arrays element-wise"""

    def __rmul__(self, rhs: object) -> Array:
        """Multiply two arrays element-wise"""

    def __truediv__(self, rhs: object) -> Array:
        """Divide two arrays element-wise"""

    def __rtruediv__(self, rhs: object) -> Array:
        """Divide two arrays element-wise"""

    def __pow__(self, rhs: object) -> Array:
        """
        Raise the left-hand side array to the power in the right-hand side array element-wise
        """

    def __iadd__(self, rhs: object) -> Array:
        """In-place add two arrays element-wise"""

    def __isub__(self, rhs: object) -> Array:
        """In-place subtract two arrays element-wise"""

    def __imul__(self, rhs: object) -> Array:
        """In-place multiply two arrays element-wise"""

    def __itruediv__(self, rhs: object) -> Array:
        """In-place divide two arrays element-wise"""

    def __ipow__(self, rhs: object) -> Array:
        """
        In-place raise the left-hand side array to the power in the right-hand side array element-wise
        """

    def __matmul__(self, rhs: Array) -> Array:
        """Matrix multiply two arrays"""

    def detach(self, arg: bool, /) -> Array:
        """Detach array from computation graph"""

    def exp(self, in_place: bool = False) -> Array:
        """Compute exponential of array elements"""

    def log(self, in_place: bool = False) -> Array:
        """Compute natural logarithm of array elements"""

    def sqrt(self, in_place: bool = False) -> Array:
        """Compute square root of array elements"""

    def sq(self, in_place: bool = False) -> Array:
        """Compute square of array elements"""

    def neg(self, in_place: bool = False) -> Array:
        """Compute negative of array elements"""

    def __neg__(self) -> Array:
        """Compute negative of array elements"""

    def recip(self, in_place: bool = False) -> Array:
        """Compute reciprocal of array elements"""

    def __eq__(self, rhs: object) -> Array:
        """Element-wise equality comparison"""

    def __ne__(self, rhs: object) -> Array:
        """Element-wise inequality comparison"""

    def __lt__(self, rhs: object) -> Array:
        """Element-wise less than comparison"""

    def __gt__(self, rhs: object) -> Array:
        """Element-wise greater than comparison"""

    def __le__(self, rhs: object) -> Array:
        """Element-wise less than or equal comparison"""

    def __ge__(self, rhs: object) -> Array:
        """Element-wise greater than or equal comparison"""

    def bitwise_not(self, in_place: bool = False) -> Array:
        """
        Element-wise bitwise not if the array type is integral and logical not if the array type is boolean
        """

    def __invert__(self) -> Array:
        """
        Element-wise bitwise not if the array type is integral and logical not if the array type is boolean
        """

    def __and__(self, rhs: object) -> Array:
        """
        Element-wise bitwise and if the array type is integral and logical and if the array type is boolean
        """

    def __or__(self, rhs: object) -> Array:
        """
        Element-wise bitwise or if the array type is integral and logical or if the array type is boolean
        """

    def __xor__(self, rhs: object) -> Array:
        """
        Element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean
        """

    def __rand__(self, rhs: object) -> Array:
        """
        Element-wise bitwise and if the array type is integral and logical and if the array type is boolean
        """

    def __ror__(self, rhs: object) -> Array:
        """
        Element-wise bitwise or if the array type is integral and logical or if the array type is boolean
        """

    def __rxor__(self, rhs: object) -> Array:
        """
        Element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean
        """

    def __iand__(self, rhs: object) -> Array:
        """
        In-place element-wise bitwise and if the array type is integral and logical and if the array type is boolean
        """

    def __ior__(self, rhs: object) -> Array:
        """
        In-place element-wise bitwise or if the array type is integral and logical or if the array type is boolean
        """

    def __ixor__(self, rhs: object) -> Array:
        """
        In-place element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean
        """

    def minimum(self, rhs: object) -> Array:
        """Element-wise minimum comparison"""

    def maximum(self, rhs: object) -> Array:
        """Element-wise maximum comparison"""

    def sum(self, dims: Sequence[int] = []) -> Array:
        """Sum array elements along specified dimensions"""

    def mean(self, dims: Sequence[int] = []) -> Array:
        """Mean value along specified dimensions"""

    def max(self, dims: Sequence[int] = []) -> Array:
        """Maximum value along specified dimensions"""

    def min(self, dims: Sequence[int] = []) -> Array:
        """Minimum value along specified dimensions"""

    def argmax(self, dims: Sequence[int] = []) -> Array:
        """Indices of maximum values along specified dimensions"""

    def argmin(self, dims: Sequence[int] = []) -> Array:
        """Indices of minimum values along specified dimensions"""

    def broadcast(self, view: Sequence[int]) -> Array:
        """Broadcast array to new shape"""

    def broadcast_to(self, view: Sequence[int]) -> Array:
        """Broadcast array to target shape"""

    def __getitem__(self, index: object) -> Array:
        """Slice array along specified dimensions"""

    def reshape(self, view: Sequence[int]) -> Array:
        """Reshape array to new dimensions"""

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Array:
        """Flatten dimensions from start to end"""

    def squeeze(self, dims: Sequence[int] = []) -> Array:
        """Remove single-dimensional entry from array"""

    def unsqueeze(self, dims: Sequence[int] = []) -> Array:
        """Add single-dimensional entry to array"""

    def permute(self, dims: Sequence[int]) -> Array:
        """Permute array dimensions"""

    def transpose(self, start_dim: int = -2, end_dim: int = -1) -> Array:
        """Transpose array dimensions"""

    def astype(self, dtype: DType) -> Array:
        """Cast array to specified dtype"""

    def eval(self) -> None:
        """Evaluate array and materialize values"""

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

    def __str__(self) -> str:
        """String representation of array"""

def full(view: Sequence[int], c: object, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array filled with specified value"""

def full_like(array: Array, c: object, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """
    Create a new array filled with specified value with same shape as the input array
    """

def zeros(view: Sequence[int], dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array filled with zeros"""

def ones(view: Sequence[int], dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array filled with ones"""

def arange(view: Sequence[int], start: int, step: int, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array with evenly spaced values"""

def zeros_like(array: Array, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array of zeros with same shape as input"""

def ones_like(array: Array, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array of ones with same shape as input"""

def from_numpy(array: NDArray, is_param: bool = False) -> Array:
    """Convert NumPy array to array"""

def uniform(view: Sequence[int], low: object = 0.0, high: object = 1.0, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array with random values from a uniform distribution"""

def normal(view: Sequence[int], mean: object = 0.0, std: object = 1.0, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array with random values from a normal distribution"""

def kaiming_uniform(view: Sequence[int], dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """
    Create a new array with random values from a Kaiming uniform distribution
    """

def randint(view: Sequence[int], low: object = 0, high: object = 10, dtype: DType = ..., device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """
    Create a new array with random integer values from a uniform distribution
    """

def randbool(view: Sequence[int], device_kind: DeviceKind = DeviceKind.MPS, device_id: int = 0, is_param: bool = False) -> Array:
    """Create a new array with uniformly distributed random boolean values"""
