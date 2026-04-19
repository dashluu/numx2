#include "functional.h"

namespace nx::core {
    Array from_buffer(uint8_t *ptr, usize nbytes, const Shape &shape, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::from_buffer(ptr, nbytes, shape, dtype, device, is_param), runtime);
    }

    Array arange(const ShapeView &view, isize start, isize step, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::arange(view, start, step, dtype, device, is_param), runtime);
    }

    Array kaiming_uniform(const ShapeView &view, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        if (!is_float(dtype)) {
            throw IncompatDTypeForRandomFunction("kaiming_uniform", "float", dtype->str());
        }

        auto [fan_in, fan_out] = compute_fan_in_and_fan_out(view);
        float low = -std::sqrt(6.0f / fan_in);
        float high = std::sqrt(6.0f / fan_in);
        return uniform(view, low, high, dtype, device_kind, device_id, is_param);
    }

    Array zeros(const ShapeView &view, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::zeros(view, dtype, device, is_param), runtime);
    }

    Array ones(const ShapeView &view, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::ones(view, dtype, device, is_param), runtime);
    }

    Array zeros_like(const Array &array, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::zeros_like(array.op(), dtype, device, is_param), runtime);
    }

    Array ones_like(const Array &array, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::ones_like(array.op(), dtype, device, is_param), runtime);
    }

    Array empty(const ShapeView &view, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::empty(view, dtype, device, is_param), runtime);
    }

    Array empty_like(const Array &array, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        Runtime *runtime = backend_runtime(device_id, device_kind);
        const Device *device = runtime->context()->device();
        return Array(graph::empty_like(array.op(), dtype, device, is_param), runtime);
    }

    std::pair<usize, usize> compute_fan_in_and_fan_out(const ShapeView &view) {
        if (view.size() < 2) {
            throw std::invalid_argument(std::format("fan-in and fan-out cannot be computed for a view of {} dimensions, which is fewer than 2.", view.size()));
        }

        usize receptive_field_size = std::accumulate(view.begin() + 2, view.end(), uone, std::multiplies<usize>());
        usize fan_in = view[1] * receptive_field_size;
        usize fan_out = view[0] * receptive_field_size;
        // Note: fan-in and fan-out cannot be 0 since each dimension > 0
        return {fan_in, fan_out};
    }

    std::pair<usize, usize> compute_fan_in_and_fan_out(const Array &array) {
        const ShapeView &view = array.view();

        if (view.size() < 2) {
            throw std::invalid_argument(std::format("fan-in and fan-out cannot be computed for array of {} dimensions, which is fewer than 2.", view.size()));
        }

        return compute_fan_in_and_fan_out(view);
    }
} // namespace nx::core
