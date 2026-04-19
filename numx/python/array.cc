#include "array.h"

namespace nx::bind {
    usize py_index(usize len, isize index) {
        isize ilen = len;

        if (index < -ilen || index >= ilen) {
            throw std::out_of_range(std::format("index {} is out of range [{}, {}).", index, -ilen, ilen));
        }

        return index < 0 ? index + ilen : index;
    }

    ShapeDims py_indices(usize len, ShapeDims &dims) {
        if (!dims.empty()) {
            std::transform(dims.begin(), dims.end(), dims.begin(), [len](auto dim) { return py_index(len, dim); });
        }

        return dims;
    }

    Range slice_to_range(usize len, const nb::object &slice) {
        // Note: no need to check for out-of-bounds indices when converting to range
        // Shape does the checking eventually
        if (!nb::isinstance<nb::slice>(slice)) {
            throw NanobindInvalidArgumentType("slice", py_class_name(slice));
        }

        auto nb_slice = nb::cast<nb::slice>(slice);
        bool start_none = nb_slice.attr("start").is_none();
        bool stop_none = nb_slice.attr("stop").is_none();
        bool step_none = nb_slice.attr("step").is_none();
        usize start, stop;
        isize step;

        if (step_none) {
            start = start_none ? 0 : py_index(len, nb::cast<usize>(nb_slice.attr("start")));
            stop = stop_none ? len : py_index(len, nb::cast<usize>(nb_slice.attr("stop")));
            return Range(start, stop, 1);
        }

        step = nb::cast<isize>(nb_slice.attr("step"));

        if (step > 0) {
            start = start_none ? 0 : py_index(len, nb::cast<usize>(nb_slice.attr("start")));
            stop = stop_none ? len : py_index(len, nb::cast<usize>(nb_slice.attr("stop")));
        } else {
            start = start_none ? len - 1 : py_index(len, nb::cast<usize>(nb_slice.attr("start")));
            // Note: this does not include the first element
            // TODO: maybe support first element inclusion?
            stop = stop_none ? 0 : py_index(len, nb::cast<usize>(nb_slice.attr("stop")));
        }

        return Range(start, stop, step);
    }

    std::vector<Range> selector_to_ranges(const Array &array, const nb::object &selector) {
        std::vector<Range> ranges;
        const Shape &shape = array.shape();

        // selector can be an int, a slice, or a sequence of ints or slices
        if (nb::isinstance<nb::int_>(selector)) {
            usize index = py_index(shape[0], nb::cast<isize>(selector));
            ranges.emplace_back(index, index + 1, 1);

            for (usize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::slice>(selector)) {
            ranges.push_back(slice_to_range(shape[0], selector));

            for (usize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::sequence>(selector) && !nb::isinstance<nb::str>(selector)) {
            // selector is a sequence but not a string
            auto sequence = nb::cast<nb::sequence>(selector);
            size_t seq_len = nb::len(sequence);

            if (seq_len > shape.get_ndim()) {
                throw std::out_of_range(std::format("index {} is out of range [{}, {}).", seq_len, 1, shape.get_ndim() + 1));
            }

            for (size_t i = 0; i < seq_len; i++) {
                auto elm = sequence[i];
                // elm must be a sequence of ints or slices
                if (nb::isinstance<nb::int_>(elm)) {
                    usize index = py_index(shape[i], nb::cast<isize>(elm));
                    ranges.emplace_back(index, index + 1, 1);
                } else if (nb::isinstance<nb::slice>(elm)) {
                    ranges.push_back(slice_to_range(shape[i], elm));
                } else {
                    throw NanobindInvalidArgumentType("int, slice", py_class_name(elm));
                }
            }

            for (usize i = seq_len; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        }

        throw NanobindInvalidArgumentType("int, slice, sequence", py_class_name(selector));
    }

    const DType *dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype) {
        if (nb_dtype == nb::dtype<float>()) {
            return &f32;
        } else if (nb_dtype == nb::dtype<int>()) {
            return &i32;
        } else if (nb_dtype == nb::dtype<bool>()) {
            return &b8;
        }

        throw nb::type_error("nanobind data type cannot be converted to numx data type.");
    }

    nb::ndarray<nb::numpy> array_to_numpy(Array &array) {
        const DType *dtype = array.dtype();

        if (dtype == &f32) {
            return array_to_numpy_impl<float>(array);
        } else if (dtype == &i32) {
            return array_to_numpy_impl<int>(array);
        }

        return array_to_numpy_impl<bool>(array);
    }

    Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr, bool is_param) {
        ShapeView view;
        ShapeStride stride;

        for (size_t i = 0; i < ndarr.ndim(); ++i) {
            view.emplace_back(ndarr.shape(i));
            stride.emplace_back(ndarr.stride(i));
        }

        Shape shape(0, view, stride);
        uint8_t *ptr = reinterpret_cast<uint8_t *>(ndarr.data());
        const DType *dtype = dtype_from_nb_dtype(ndarr.dtype());
        return nxc::from_buffer(ptr, ndarr.nbytes(), shape, dtype, default_device_kind, default_device_id, is_param);
    }

    nb::ndarray<nb::pytorch> array_to_torch(Array &array) {
        const DType *dtype = array.dtype();

        if (dtype == &f32) {
            return array_to_torch_impl<float>(array);
        } else if (dtype == &i32) {
            return array_to_torch_impl<int>(array);
        }

        return array_to_torch_impl<bool>(array);
    }

    nb::object item(Array &array) {
        usize value = array.item();
        const DType *dtype = array.dtype();

        if (dtype == &f32) {
            return nb::cast<float>(std::bit_cast<float>(static_cast<int32_t>(value)));
        } else if (dtype == &i32) {
            return nb::cast<int>(value);
        }

        return nb::cast<bool>(value);
    }

    Array full(const ShapeView &view, const nb::object &constant, const DType *dtype, DeviceKind device_kind, usize device_id, bool is_param) {
        if (nb::isinstance<nb::float_>(constant)) {
            return nxc::full(view, nb::cast<float>(constant), dtype, device_kind, device_id, is_param);
        } else if (nb::isinstance<nb::int_>(constant)) {
            return nxc::full(view, nb::cast<int>(constant), dtype, device_kind, device_id, is_param);
        } else if (nb::isinstance<nb::bool_>(constant)) {
            return nxc::full(view, nb::cast<bool>(constant), dtype, device_kind, device_id, is_param);
        }

        throw NanobindInvalidArgumentType("float, int, bool", py_class_name(constant));
    }
} // namespace nx::bind
