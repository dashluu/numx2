#pragma once

#include "array.h"

namespace nx::core {
    using foundation::b8;
    using foundation::i32;
    using foundation::IncompatDTypeForRandomFunction;
    using foundation::IntegerType;
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
        return Array(graph::full(view, constant, dtype, device, is_param), runtime);
    }

    template <NumericOrBoolType T>
    Array full_like(const Array &array, T constant, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::full_like(array.op(), constant, dtype, device, is_param), runtime);
    }

    inline Array arange(const ShapeView &view, isize start, isize step, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::arange(view, start, step, dtype, device, is_param), runtime);
    }

    template <NumericType T>
    Array uniform(const ShapeView &view, T low = T(0), T high = T(1), const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        if (!is_float(dtype)) {
            throw IncompatDTypeForRandomFunction("uniform", "float", dtype->str());
        }

        Runtime *runtime = backend_runtime(device_id, device_kind);
        const RuntimeContext *runtime_ctx = runtime->context();
        const Device *device = runtime_ctx->device();
        RandomKeyGenerator *key_gen = runtime_ctx->key_generator();
        return Array(graph::uniform(view, key_gen, low, high, dtype, device, is_param), runtime);
    }

    template <NumericType T>
    Array normal(const ShapeView &view, T mean = T(0), T std = T(1), const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        if (!is_float(dtype)) {
            throw IncompatDTypeForRandomFunction("normal", "float", dtype->str());
        }

        Runtime *runtime = backend_runtime(device_id, device_kind);
        const RuntimeContext *runtime_ctx = runtime->context();
        const Device *device = runtime_ctx->device();
        RandomKeyGenerator *key_gen = runtime_ctx->key_generator();
        return Array(graph::normal(view, key_gen, mean, std, dtype, device, is_param), runtime);
    }

    Array kaiming_uniform(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false);

    template <IntegerType T>
    Array randint(const ShapeView &view, T low = T(0), T high = T(10), const DType *dtype = &i32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        if (!is_int(dtype)) {
            throw IncompatDTypeForRandomFunction("randint", "int", dtype->str());
        }

        const DType *float_dtype = float_dtype_by_dtype(dtype);
        return uniform(view, low, high, float_dtype, device_kind, device_id, is_param).astype(dtype);
    }

    inline Array randbool(const ShapeView &view, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        // TODO: change i32 to something else?
        return randint<int>(view, 0, 2, &i32, device_kind, device_id, is_param).astype(&b8);
    }

    inline Array zeros(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::zeros(view, dtype, device, is_param), runtime);
    }

    inline Array ones(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::ones(view, dtype, device, is_param), runtime);
    }

    inline Array zeros_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::zeros_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array ones_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::ones_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array empty(const ShapeView &view, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::empty(view, dtype, device, is_param), runtime);
    }

    inline Array empty_like(const Array &array, const DType *dtype = &f32, DeviceKind device_kind = DeviceKind::MPS, usize device_id = 0, bool is_param = false) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::empty_like(array.op(), dtype, device, is_param), runtime);
    }

    inline Array empty_like(const Array &array, bool is_param = false) { return Array(graph::empty_like(array.op(), is_param)); }
    std::pair<usize, usize> compute_fan_in_and_fan_out(const ShapeView &view);
    std::pair<usize, usize> compute_fan_in_and_fan_out(const Array &array);
} // namespace nx::core
