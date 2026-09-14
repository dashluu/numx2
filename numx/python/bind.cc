#include "functional.h"
#include "module.h"
#include "optim.h"

NB_MODULE(nx, m) {
    auto m_core = m.def_submodule("core", "Core module");
    auto m_nn = m.def_submodule("nn", "Neural network module");
    auto m_optim = m.def_submodule("optim", "Optimization module");

    // DType class and operations
    nb::enum_<nxf::DTypeKind>(m_core, "DTypeKind")
        .value("BOOL", nxf::DTypeKind::Bool)
        .value("SIGNED", nxf::DTypeKind::Signed)
        .value("UNSIGNED", nxf::DTypeKind::Unsigned)
        .value("FLOAT", nxf::DTypeKind::Float);

    nb::class_<nxf::DType>(m_core, "DType")
        .def_prop_ro("name", &nxf::DType::str, "Get data type's name as string")
        .def_prop_ro("size", &nxf::DType::size, "Get data type's size in bytes")
        .def_prop_ro("category", &nxf::DType::kind, "Get data type's category");

    nb::class_<nxf::F32, nxf::DType>(m_core, "F32", "32-bit floating point dtype");
    nb::class_<nxf::I32, nxf::DType>(m_core, "I32", "32-bit integer dtype");
    nb::class_<nxf::B8, nxf::DType>(m_core, "B8", "Boolean dtype");

    // Global dtype instances
    m_core.attr("f32") = &nxf::f32;
    m_core.attr("i32") = &nxf::i32;
    m_core.attr("b8") = &nxf::b8;

    // Shape class
    nb::class_<nxf::Shape>(m_core, "Shape")
        .def_prop_ro("offset", &nxf::Shape::get_offset, "Get shape's offset")
        .def_prop_ro("view", &nxf::Shape::get_view, "Get shape's view")
        .def_prop_ro("stride", &nxf::Shape::get_stride, "Get shape's stride")
        .def_prop_ro("ndim", &nxf::Shape::get_ndim, "Get shape's number of dimensions")
        .def_prop_ro("numel", &nxf::Shape::get_numel, "Get shape's total number of elements")
        .def("__str__", &nxf::Shape::str, "String representation of shape");

    // Device class
    nb::enum_<nxf::DeviceKind>(m_core, "DeviceKind")
        .value("CPU", nxf::DeviceKind::CPU)
        .value("MPS", nxf::DeviceKind::MPS);

    nb::class_<nxf::Device>(m_core, "Device")
        .def_prop_ro("kind", &nxf::Device::kind, "Get device's kind")
        .def_prop_ro("id", &nxf::Device::id, "Get device's ID")
        .def("__str__", &nxf::Device::str, "String representation of device");

    // Array class
    nb::class_<nxc::Array>(m_core, "Array")
        // Properties
        .def_prop_ro("shape", &nxc::Array::shape, "Get array's shape")
        .def_prop_ro("dtype", &nxc::Array::dtype, "Get array's data type")
        .def_prop_ro("device", &nxc::Array::device, "Get array's device")
        .def_prop_ro("grad", &nxc::Array::grad, "Get array's gradient")
        .def_prop_ro("ndim", &nxc::Array::ndim, "Get array's number of dimensions")
        .def_prop_ro("numel", &nxc::Array::numel, "Get array's total number of elements")
        .def_prop_ro("offset", &nxc::Array::offset, "Get array's offset")
        .def_prop_ro("view", &nxc::Array::view, "Get array's view")
        .def_prop_ro("stride", &nxc::Array::stride, "Get array's stride")
        .def_prop_ro("ptr", &nxc::Array::ptr, "Get array's raw data pointer")
        .def_prop_ro("itemsize", &nxc::Array::itemsize, "Get array's element size in bytes")
        .def_prop_ro("nbytes", &nxc::Array::nbytes, "Get array's total size in bytes")
        .def("size", &nxc::Array::size, "dim"_a, "Get array's number of elements in certain dimension")
        .def("__len__", [](const nxc::Array &array) { return array.size(0); }, "Get array's length, that is, the number of elements in the first dimension")
        .def_prop_ro("is_parameter", &nxc::Array::is_parameter, "Check if array is a parameter")
        .def_prop_ro("is_contiguous", &nxc::Array::is_contiguous, "Check if array is contiguous")
        .def_prop_rw("grad_enabled", &nxc::Array::is_grad_enabled, &nxc::Array::enable_grad, "enabled"_a, "Get/set array's gradient tracking, setter can only be used before compilation and forwarding, otherwise, there is no effect")

        // N-dimensional array
        .def("numpy", &nxb::array_to_numpy, nb::rv_policy::reference_internal, "Convert array to NumPy array")
        .def("torch", &nxb::array_to_torch, nb::rv_policy::reference_internal, "Convert array to PyTorch tensor")
        // .def_static("from_torch", &nxb::array_from_torch, "tensor"_a, "Convert PyTorch tensor to array")
        .def("item", &nxb::item, "Get array's only value")
        // .def("graph", &nxc::Array::graph_str, "Get array's computation graph representation")

        // Element-wise operations
        .def("__add__", &nxb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__radd__", &nxb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__sub__", &nxb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__rsub__", &nxb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__mul__", &nxb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__rmul__", &nxb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__truediv__", &nxb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__rtruediv__", &nxb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__pow__", &nxb::pow, "rhs"_a, "Raise the left-hand side array to the power in the right-hand side array element-wise")
        .def("__iadd__", &nxb::i_add, "rhs"_a, "In-place add two arrays element-wise")
        .def("__isub__", &nxb::i_sub, "rhs"_a, "In-place subtract two arrays element-wise")
        .def("__imul__", &nxb::i_mul, "rhs"_a, "In-place multiply two arrays element-wise")
        .def("__itruediv__", &nxb::i_div, "rhs"_a, "In-place divide two arrays element-wise")
        .def("__ipow__", &nxb::i_pow, "rhs"_a, "In-place raise the left-hand side array to the power in the right-hand side array element-wise")
        .def("__matmul__", &nxc::Array::gemm, "rhs"_a, "Matrix multiply two arrays")
        .def("detach", &nxc::Array::detach, "Detach array from computation graph")
        .def("exp", &nxc::Array::exp, "in_place"_a = false, "Compute exponential of array elements")
        .def("log", &nxc::Array::log, "in_place"_a = false, "Compute natural logarithm of array elements")
        .def("sqrt", &nxc::Array::sqrt, "in_place"_a = false, "Compute square root of array elements")
        .def("sq", &nxc::Array::sq, "in_place"_a = false, "Compute square of array elements")
        .def("neg", &nxc::Array::neg, "in_place"_a = false, "Compute negative of array elements")
        .def("__neg__", &nxb::neg, "Compute negative of array elements")
        .def("recip", &nxc::Array::recip, "in_place"_a = false, "Compute reciprocal of array elements")

        // Comparison operations
        .def("__eq__", &nxb::eq, "rhs"_a, "Element-wise equality comparison")
        .def("__ne__", &nxb::neq, "rhs"_a, "Element-wise inequality comparison")
        .def("__lt__", &nxb::less, "rhs"_a, "Element-wise less than comparison")
        .def("__gt__", &nxb::greater, "rhs"_a, "Element-wise greater than comparison")
        .def("__le__", &nxb::leq, "rhs"_a, "Element-wise less than or equal comparison")
        .def("__ge__", &nxb::geq, "rhs"_a, "Element-wise greater than or equal comparison")
        .def("bitwise_not", &nxc::Array::bitwise_not, "in_place"_a = false, "Element-wise bitwise not if the array type is integral and logical not if the array type is boolean")
        .def("__invert__", &nxb::bitwise_logic_not, "Element-wise bitwise not if the array type is integral and logical not if the array type is boolean")
        .def("__and__", &nxb::bitwise_and, "rhs"_a, "Element-wise bitwise and if the array type is integral and logical and if the array type is boolean")
        .def("__or__", &nxb::bitwise_or, "rhs"_a, "Element-wise bitwise or if the array type is integral and logical or if the array type is boolean")
        .def("__xor__", &nxb::bitwise_xor, "rhs"_a, "Element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean")
        .def("__rand__", &nxb::bitwise_and, "rhs"_a, "Element-wise bitwise and if the array type is integral and logical and if the array type is boolean")
        .def("__ror__", &nxb::bitwise_or, "rhs"_a, "Element-wise bitwise or if the array type is integral and logical or if the array type is boolean")
        .def("__rxor__", &nxb::bitwise_xor, "rhs"_a, "Element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean")
        .def("__iand__", &nxb::i_bitwise_and, "rhs"_a, "In-place element-wise bitwise and if the array type is integral and logical and if the array type is boolean")
        .def("__ior__", &nxb::i_bitwise_or, "rhs"_a, "In-place element-wise bitwise or if the array type is integral and logical or if the array type is boolean")
        .def("__ixor__", &nxb::i_bitwise_xor, "rhs"_a, "In-place element-wise bitwise xor if the array type is integral and logical xor if the array type is boolean")
        .def("minimum", &nxb::minimum, "rhs"_a, "Element-wise minimum comparison")
        .def("maximum", &nxb::maximum, "rhs"_a, "Element-wise maximum comparison")

        // Reduction operations
        .def("sum", &nxb::sum, "dims"_a = nxf::ShapeDims{}, "Sum array elements along specified dimensions")
        .def("mean", &nxb::mean, "dims"_a = nxf::ShapeDims{}, "Mean value along specified dimensions")
        .def("max", &nxb::max, "dims"_a = nxf::ShapeDims{}, "Maximum value along specified dimensions")
        .def("min", &nxb::min, "dims"_a = nxf::ShapeDims{}, "Minimum value along specified dimensions")
        .def("argmax", &nxb::argmax, "dims"_a = nxf::ShapeDims{}, "Indices of maximum values along specified dimensions")
        .def("argmin", &nxb::argmin, "dims"_a = nxf::ShapeDims{}, "Indices of minimum values along specified dimensions")

        // Shape operations
        .def("broadcast", &nxc::Array::broadcast, "view"_a, "Broadcast array to new shape")
        .def("broadcast_to", &nxc::Array::broadcast_to, "view"_a, "Broadcast array to target shape")
        .def("__getitem__", &nxb::slice, "index"_a, "Slice array along specified dimensions")
        .def("reshape", &nxc::Array::reshape, "view"_a, "Reshape array to new dimensions")
        .def("flatten", &nxb::flatten, "start_dim"_a = 0, "end_dim"_a = -1, "Flatten dimensions from start to end")
        .def("squeeze", &nxb::squeeze, "dims"_a = nxf::ShapeDims{}, "Remove single-dimensional entry from array")
        .def("unsqueeze", &nxb::unsqueeze, "dims"_a = nxf::ShapeDims{}, "Add single-dimensional entry to array")
        .def("permute", &nxb::permute, "dims"_a, "Permute array dimensions")
        .def("transpose", &nxb::transpose, "start_dim"_a = -2, "end_dim"_a = -1, "Transpose array dimensions")

        // Type operations
        .def("astype", &nxc::Array::astype, "dtype"_a, "Cast array to specified dtype")

        // Evaluation and backward
        .def("eval", &nxc::Array::eval, "Evaluate array and materialize values")
        .def("backward", &nxc::Array::backward, "Compute gradients through backpropagation")

        // String representation
        .def("__str__", &nxc::Array::str, "String representation of array");

    m_core.def("full", &nxb::full, "view"_a, "c"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array filled with specified value")
        .def("full_like", &nxb::full_like, "array"_a, "c"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array filled with specified value with same shape as the input array")
        .def("zeros", &nxc::zeros, "view"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array filled with zeros")
        .def("ones", &nxc::ones, "view"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array filled with ones")
        .def("arange", &nxc::arange, "view"_a, "start"_a, "step"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with evenly spaced values")
        .def("zeros_like", &nxc::zeros_like, "array"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array of zeros with same shape as input")
        .def("ones_like", &nxc::ones_like, "array"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array of ones with same shape as input")
        .def("from_numpy", &nxb::array_from_numpy, "array"_a, "is_param"_a = false, "Convert NumPy array to array")
        .def("uniform", &nxb::uniform, "view"_a, "low"_a = 0.0, "high"_a = 1.0, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with random values from a uniform distribution")
        .def("normal", &nxb::normal, "view"_a, "mean"_a = 0.0, "std"_a = 1.0, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with random values from a normal distribution")
        .def("kaiming_uniform", &nxc::kaiming_uniform, "view"_a, "dtype"_a = &nxf::f32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with random values from a Kaiming uniform distribution")
        .def("randint", &nxb::randint, "view"_a, "low"_a = 0, "high"_a = 10, "dtype"_a = &nxf::i32, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with random integer values from a uniform distribution")
        .def("randbool", &nxb::randbool, "view"_a, "device_kind"_a = nxf::default_device_kind, "device_id"_a = nxf::default_device_id, "is_param"_a = false, "Create a new array with uniformly distributed random boolean values");

    m_nn.def("linear", &nxn::linear, "x"_a, "weight"_a, "Functional linear without bias");
    m_nn.def("linear_with_bias", &nxn::linear_with_bias, "x"_a, "weight"_a, "bias"_a, "Functional linear with bias");
    m_nn.def("relu", &nxn::relu, "x"_a, "ReLU activation function");
    m_nn.def("onehot", &nxb::onehot, "x"_a, "num_classes"_a = -1, "One-hot encode input array");
    m_nn.def("softmax", &nxb::softmax, "x"_a, "dim"_a = -1, "Compute softmax for input array");
    m_nn.def("cross_entropy_loss", &nxn::cross_entropy_loss, "x"_a, "y"_a, "Compute cross-entropy loss between input x and target y");

    nb::class_<nxn::Module, nxb::PyModule>(m_nn, "Module")
        .def(nb::init(), "Base module")
        .def("add_parameter", &nxb::ModulePublicist::add_parameter, "param"_a, "Add parameter to module")
        // Tell Python not to free the objects in vector since they are pointers to valid objects inside Module
        // and Python does not know how to free it any way
        .def("parameters", &nxn::Module::parameters, nb::rv_policy::reference_internal, "Get module parameters")
        .def("forward", &nxn::Module::forward, "x"_a, "Forward pass through module")
        .def("__call__", &nxn::Module::operator(), "x"_a, "Forward pass through module");

    nb::class_<nxn::Linear, nxn::Module>(m_nn, "Linear")
        .def(nb::init<nxc::usize, nxc::usize, bool>(), "in_features"_a, "out_features"_a, "bias"_a = true, "Linear layer")
        .def_prop_ro("weight", &nxn::Linear::weight, "Get linear layer weight")
        .def_prop_ro("bias", &nxn::Linear::bias, "Get linear layer bias");

    nb::class_<nxo::Optimizer, nxb::PyOptimizer>(m_optim, "Optimizer")
        .def(nb::init<float>(), "lr"_a, "Base optimizer")
        .def("init_single", &nxo::Optimizer::init_single, "param"_a, "state"_a, "Initialize state for a single parameter")
        .def("apply_single", &nxo::Optimizer::apply_single, "param"_a, "grad"_a, "state"_a, "Update function for a single parameter")
        .def("update", &nxo::Optimizer::update, "params"_a, "states"_a, "Update module parameters");

    nb::class_<nxo::SGD, nxo::Optimizer>(m_optim, "SGD")
        .def(nb::init<float, float, float, float, bool>(), "lr"_a, "momentum"_a = 0.0, "weight_decay"_a = 0.0, "dampening"_a = 0.0, "nesterov"_a = false, "Stochastic Gradient Descent optimizer");
    nb::class_<nxo::Adam, nxo::Optimizer>(m_optim, "Adam")
        .def(nb::init<float, float, float, float, bool>(), "lr"_a, "beta1"_a = 0.9, "beta2"_a = 0.999, "epsilon"_a = 1e-8, "bias_correction"_a = false, "Adam optimizer");
}
