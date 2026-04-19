#pragma once

#include "bind.h"

namespace nx::bind {
    using core::Array;
    using foundation::b8;
    using foundation::default_device_id;
    using foundation::default_device_kind;
    using foundation::DeviceKind;
    using foundation::DType;
    using foundation::f32;
    using foundation::i32;
    using foundation::isize;
    using foundation::NanobindInvalidArgumentType;
    using foundation::Range;
    using foundation::Shape;
    using foundation::ShapeDims;
    using foundation::ShapeStride;
    using foundation::ShapeView;
    using foundation::usize;

    template <class T>
    nb::ndarray<nb::numpy> array_to_numpy_impl(Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);

        return nb::ndarray<nb::numpy>(
            array.ptr(),
            array.ndim(),
            array.view().data(),
            pyarr.ptr(),
            array.stride().data(),
            nb::dtype<T>(),
            // Numpy can only run on the cpu
            nb::device::cpu::value,
            'C');
    }

    template <class T>
    nb::ndarray<nb::pytorch> array_to_torch_impl(Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);
        int device;

        switch (array.device()->kind()) {
        case DeviceKind::CPU:
            device = nb::device::cpu::value;
            break;
        default:
            // Try CPU for now
            // TODO: change to metal later
            device = nb::device::cpu::value;
            break;
        }

        return nb::ndarray<nb::pytorch>(
            array.ptr(),
            array.ndim(),
            array.view().data(),
            pyarr.ptr(),
            array.stride().data(),
            nb::dtype<T>(),
            device,
            'C');
    }

    inline std::string py_class_name(const nb::object &py_obj) {
        auto cls = py_obj.attr("__class__");
        auto name = cls.attr("__name__");
        return nb::cast<std::string>(name);
    }

    template <class F>
    Array binary(const Array &array, const nb::object &rhs, F &&f) {
        if (nb::isinstance<Array>(rhs)) {
            return f(array, nb::cast<Array>(rhs));
        } else if (nb::isinstance<nb::float_>(rhs)) {
            return f(array, nb::cast<float>(rhs));
        } else if (nb::isinstance<nb::int_>(rhs)) {
            return f(array, nb::cast<int>(rhs));
        } else if (nb::isinstance<nb::bool_>(rhs)) {
            return f(array, nb::cast<bool>(rhs));
        }

        throw nxf::NanobindInvalidArgumentType("float, int, bool, Array", py_class_name(rhs));
    }

    template <class F>
    Array in_place_binary(Array &array, const nb::object &rhs, F &&f) {
        if (nb::isinstance<Array>(rhs)) {
            return f(array, nb::cast<Array>(rhs));
        } else if (nb::isinstance<nb::float_>(rhs)) {
            return f(array, nb::cast<float>(rhs));
        } else if (nb::isinstance<nb::int_>(rhs)) {
            return f(array, nb::cast<int>(rhs));
        } else if (nb::isinstance<nb::bool_>(rhs)) {
            return f(array, nb::cast<bool>(rhs));
        }

        throw nxf::NanobindInvalidArgumentType("float, int, bool, Array", py_class_name(rhs));
    }

    usize py_index(usize len, isize index);
    ShapeDims py_indices(usize len, ShapeDims &dims);
    Range slice_to_range(usize len, const nb::object &slice);
    std::vector<Range> selector_to_ranges(const Array &array, const nb::object &selector);
    const DType *dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
    nb::ndarray<nb::numpy> array_to_numpy(Array &array);
    Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr, bool is_param = false);
    nb::ndarray<nb::pytorch> array_to_torch(Array &array);
    nb::object item(Array &array);
    Array full(const ShapeView &view, const nb::object &constant, const DType *dtype, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false);

    inline Array full_like(const Array &array, const nb::object &constant, const DType *dtype, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false) {
        return full(array.view(), constant, dtype, device_kind, device_id, is_param);
    }

    inline Array uniform(const ShapeView &view, const nb::object &low, const nb::object &high, const DType *dtype, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false) {
        return nxc::uniform(view, nb::cast<float>(low), nb::cast<float>(high), dtype, device_kind, device_id, is_param);
    }

    inline Array normal(const ShapeView &view, const nb::object &mean, const nb::object &std, const DType *dtype, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false) {
        return nxc::normal(view, nb::cast<float>(mean), nb::cast<float>(std), dtype, device_kind, device_id, is_param);
    }

    inline Array randint(const ShapeView &view, const nb::object &low, const nb::object &high, const DType *dtype, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false) {
        return nxc::randint(view, nb::cast<int>(low), nb::cast<int>(high), dtype, device_kind, device_id, is_param);
    }

    inline Array randbool(const ShapeView &view, DeviceKind device_kind = default_device_kind, usize device_id = default_device_id, bool is_param = false) {
        return nxc::randbool(view, device_kind, device_id, is_param);
    }

    inline Array neg(const Array &array) { return array.neg(); }

    inline Array add(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a + b; });
    }

    inline Array i_add(Array &array, const nb::object &rhs) {
        return in_place_binary(array, rhs, [](auto &a, const auto &b) { return a += b; });
    }

    inline Array sub(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a - b; });
    }

    inline Array i_sub(Array &array, const nb::object &rhs) {
        return in_place_binary(array, rhs, [](auto &a, const auto &b) { return a -= b; });
    }

    inline Array mul(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a * b; });
    }

    inline Array i_mul(Array &array, const nb::object &rhs) {
        return in_place_binary(array, rhs, [](auto &a, const auto &b) { return a *= b; });
    }

    inline Array div(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a / b; });
    }

    inline Array i_div(Array &array, const nb::object &rhs) {
        return in_place_binary(array, rhs, [](auto &a, const auto &b) { return a /= b; });
    }

    inline Array eq(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a == b; });
    }

    inline Array neq(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a != b; });
    }

    inline Array less(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a < b; });
    }

    inline Array greater(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a > b; });
    }

    inline Array leq(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a <= b; });
    }

    inline Array geq(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a >= b; });
    }

    inline Array minimum(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a.minimum(b); });
    }

    inline Array maximum(const Array &array, const nb::object &rhs) {
        return binary(array, rhs, [](const auto &a, const auto &b) { return a.maximum(b); });
    }

    inline Array slice(const Array &array, const nb::object &selector) {
        return array.slice(selector_to_ranges(array, selector));
    }

    inline Array permute(const Array &array, ShapeDims &dims) {
        return array.permute(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array transpose(const Array &array, isize start_dim, isize end_dim) {
        return array.transpose(py_index(array.shape().get_ndim(), start_dim), py_index(array.shape().get_ndim(), end_dim));
    }

    inline Array flatten(const Array &array, isize start_dim, isize end_dim) {
        return array.flatten(py_index(array.shape().get_ndim(), start_dim), py_index(array.shape().get_ndim(), end_dim));
    }

    inline Array squeeze(const Array &array, ShapeDims &dims) {
        return array.squeeze(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array unsqueeze(const Array &array, ShapeDims &dims) {
        return array.unsqueeze(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array sum(const Array &array, ShapeDims &dims) {
        return array.sum(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array mean(const Array &array, ShapeDims &dims) {
        return array.mean(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array max(const Array &array, ShapeDims &dims) {
        return array.max(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array min(const Array &array, ShapeDims &dims) {
        return array.min(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array argmax(const Array &array, ShapeDims &dims) {
        return array.argmax(py_indices(array.shape().get_ndim(), dims));
    }

    inline Array argmin(const Array &array, ShapeDims &dims) {
        return array.argmin(py_indices(array.shape().get_ndim(), dims));
    }
} // namespace nx::bind
