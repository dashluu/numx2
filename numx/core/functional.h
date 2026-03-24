#pragma once

#include "array.h"

namespace nx::core {
    using foundation::uone;

    template <NumericType T>
    Array operator+(T constant, const Array &array) { return array + constant; }

    template <NumericType T>
    Array operator-(T constant, const Array &array) { return array - constant; }

    template <NumericType T>
    Array operator*(T constant, const Array &array) { return array * constant; }

    template <NumericType T>
    Array operator/(T constant, const Array &array) { return array.recip() * constant; }

    inline Array from_buffer(uint8_t *ptr, usize nbytes, const Shape &shape, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::from_buffer(ptr, nbytes, shape, dtype, device, is_param), runtime);
    }

    template <NumericOrBoolType T>
    Array full(const ShapeView &view, T constant, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::full(view, constant, dtype, device, is_param), runtime);
    }

    template <NumericOrBoolType T>
    Array full_like(const Array &array, T constant, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::full_like(array.op(), constant, dtype, device, is_param), runtime);
    }

    inline Array arange(const ShapeView &view, isize start, isize step, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::arange(view, start, step, dtype, device, is_param), runtime);
    }

    inline Array zeros(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::zeros(view, dtype, device, is_param), runtime);
    }

    inline Array ones(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::ones(view, dtype, device, is_param), runtime);
    }

    inline Array zeros_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::zeros_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array ones_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::ones_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array empty(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::empty(view, dtype, device, is_param), runtime);
    }

    inline Array empty_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(nx::graph::empty_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array empty_like(const Array &array) { return Array(nx::graph::empty_like(array.op())); }
    std::pair<usize, usize> compute_fan_in_and_fan_out(const ShapeView &view);
    std::pair<usize, usize> compute_fan_in_and_fan_out(const Array &array);
} // namespace nx::core
