#include "shape.h"

namespace nx::foundation {
    void Shape::are_ranges_valid(const RangeVec &ranges) const {
        if (ranges.size() != get_ndim()) {
            throw std::invalid_argument(std::format("the number of ranges does not match the number of dimensions: {} and {}.", ranges.size(), get_ndim()));
        }

        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const Range range = ranges[i];

            if (range.start() >= m_view[i]) {
                throw std::invalid_argument(std::format("start {} is not in the range [0, {}).", range.start(), m_view[i]));
            }

            if (range.stop() > m_view[i]) {
                throw std::invalid_argument(std::format("stop {} is not in the range [0, {}].", range.stop(), m_view[i]));
            }

            if (range.step() == 0) {
                throw std::invalid_argument("step cannot be 0.");
            }

            if (range.start() < range.stop() && range.step() < 0) {
                throw std::invalid_argument(std::format("step {} is not positive when start {} < stop {}.", range.step(), range.start(), range.stop()));
            }

            if (range.start() > range.stop() && range.step() > 0) {
                throw std::invalid_argument(std::format("step {} is not negative when start {} > stop {}.", range.step(), range.start(), range.stop()));
            }
        }
    }

    void Shape::is_dim_permutation_valid(const ShapeDims &dims) const {
        usize ndim = get_ndim();

        if (dims.size() != ndim) {
            throw std::invalid_argument(std::format("the number of dimensions in the specified order does not match the number of dimensions in the shape: {} and {}.", dims.size(), ndim));
        }

        std::vector<bool> dims_used(ndim, false);

        for (auto dim : dims) {
            if (dim >= ndim) {
                throw std::invalid_argument(std::format("the dimension must be in the range [0, {}) but got {}.", ndim, dim));
            }

            dims_used[dim] = true;
        }

        for (auto dim_used : dims_used) {
            if (!dim_used) {
                throw std::invalid_argument(std::format("the specified order must be a permutation of the dimensions but got {}.", join_nums(dims)));
            }
        }
    }

    void Shape::are_start_end_dim_valid(usize start_dim, usize end_dim) const {
        usize ndim = get_ndim();

        if (start_dim > end_dim) {
            throw std::invalid_argument("the start dimension must be smaller than the end dimension.");
        }

        if (start_dim >= ndim) {
            throw std::invalid_argument(std::format("the start dimension must be in the range [0, {}) but got {}.", ndim, start_dim));
        }

        if (end_dim >= ndim) {
            throw std::invalid_argument(std::format("the end dimension must be in the range [0, {}) but got {}.", ndim, end_dim));
        }
    }

    void Shape::is_view_valid(const ShapeView &view) {
        if (view.size() == 0) {
            throw std::invalid_argument("shape must have at least 1 dimension.");
        }

        if (std::any_of(view.begin(), view.end(), [](usize v) { return v == 0; })) {
            throw std::invalid_argument("dimension cannot be 0.");
        }
    }

    Shape::Shape(usize offset, const ShapeView &view, const ShapeStride &stride) {
        is_view_valid(view);

        if (view.size() != stride.size()) {
            throw std::invalid_argument(std::format("view {} is not compatible with stride {}.", view.size(), stride.size()));
        }

        m_offset = offset;
        m_view = view;
        m_stride = stride;
    }

    Shape::Shape(usize offset, const ShapeView &view) {
        is_view_valid(view);
        m_offset = offset;
        m_view = view;
        m_stride.resize(view.size());
        usize s = 1;

        for (std::ptrdiff_t i = view.size() - 1; i >= 0; --i) {
            m_stride[i] = s;
            s *= view[i];
        }
    }

    ShapeStride Shape::contiguous_stride() const {
        usize ndim = get_ndim();
        ShapeStride result(ndim, 0);
        isize s = 1;

        for (isize i = ndim - 1; i >= 0; --i) {
            result[i] = s;
            s *= m_view[i];
        }

        return result;
    }

    std::vector<usize> Shape::size_per_dim() const {
        usize ndim = get_ndim();
        std::vector<usize> result(ndim, 0);
        usize n = 1;

        for (isize i = ndim - 1; i >= 0; --i) {
            n *= m_view[i];
            result[i] = n;
        }

        return result;
    }

    usize Shape::get_size(usize dim) const {
        usize ndim = get_ndim();

        if (dim >= ndim) {
            throw std::invalid_argument(std::format("dimension {} is out of range [0, {}).", dim, ndim));
        }

        return m_view[dim];
    }

    bool Shape::broadcastable(const ShapeView &view) const {
        is_view_valid(view);

        if (m_view == view) {
            return true;
        }

        for (auto l_iter = m_view.rbegin(), r_iter = view.rbegin();
             l_iter != m_view.rend() && r_iter != view.rend();
             l_iter++, r_iter++) {
            if (*l_iter != *r_iter && *l_iter != 1 && *r_iter != 1) {
                return false;
            }
        }

        return true;
    }

    // One-direction broadcast check
    bool Shape::broadcastable_to(const ShapeView &view) const {
        is_view_valid(view);

        if (m_view == view) {
            return true;
        }

        if (get_ndim() > view.size()) {
            return false;
        }

        for (auto l_iter = m_view.rbegin(), r_iter = view.rbegin();
             l_iter != m_view.rend();
             l_iter++, r_iter++) {
            if (*l_iter != *r_iter && *l_iter != 1) {
                return false;
            }
        }

        return true;
    }

    bool Shape::gemm_broadcastable(const ShapeView &view) const {
        is_view_valid(view);
        usize l_ndim = get_ndim();
        usize r_ndim = view.size();

        if (l_ndim == 1) {
            if (r_ndim == 1) {
                // gevv
                return m_view[0] == view[0];
            }

            // gevm
            return m_view[0] == view[r_ndim - 2];
        } else if (r_ndim == 1) {
            // gemv
            return m_view[l_ndim - 1] == view[0];
        }

        // gemm or matrix multiplication for 2D and above
        if (m_view[l_ndim - 1] != view[view.size() - 2]) {
            return false;
        }

        for (auto l_iter = m_view.begin(), r_iter = view.begin();
             l_iter != m_view.end() - 2 && r_iter != view.end() - 2;
             l_iter++, r_iter++) {
            if (*l_iter != *r_iter && *l_iter != 1 && *r_iter != 1) {
                return false;
            }
        }

        return true;
    }

    // One-direction broadcast
    std::pair<Shape, ShapeDims> Shape::broadcast_to(const ShapeView &view) const {
        ShapeDims broadcast_dims;

        if (m_view == view) {
            return std::make_pair(*this, broadcast_dims);
        }

        if (!broadcastable_to(view)) {
            throw std::invalid_argument(std::format("cannot broadcast shape ({}) to ({}).", join_nums(m_view), join_nums(view)));
        }

        ShapeView broadcast_view = m_view;
        std::size_t ndim_diff = view.size() - broadcast_view.size();
        broadcast_view.insert(broadcast_view.begin(), ndim_diff, 1);
        Shape broadcast_shape(m_offset, broadcast_view);
        std::fill_n(broadcast_shape.m_stride.begin(), ndim_diff, 0);

        for (std::size_t i = 0; i < view.size(); ++i) {
            if (broadcast_view[i] < view[i]) {
                broadcast_dims.emplace_back(i);
                broadcast_shape.m_view[i] = view[i];
                broadcast_shape.m_stride[i] = 0;
            }
        }

        return {broadcast_shape, broadcast_dims};
    }

    // ShapeDims specifies which dimensions are broadcasted
    std::pair<Shape, ShapeDims> Shape::broadcast(const ShapeView &view) const {
        ShapeDims broadcast_dims;

        if (m_view == view) {
            return std::make_pair(*this, broadcast_dims);
        }

        if (!broadcastable(view)) {
            throw std::invalid_argument(std::format("cannot broadcast shape ({}) and ({}).", join_nums(m_view), join_nums(view)));
        }

        ShapeView l_view = m_view;
        ShapeView r_view = view;
        std::size_t ndim = std::max(l_view.size(), r_view.size());
        std::size_t l_diff = ndim - l_view.size();
        std::size_t r_diff = ndim - r_view.size();
        l_view.insert(l_view.begin(), l_diff, 1);
        r_view.insert(r_view.begin(), r_diff, 1);
        Shape broadcast_shape(m_offset, l_view);
        std::fill_n(broadcast_shape.m_stride.begin(), l_diff, 0);

        for (usize i = 0; i < ndim; ++i) {
            if (l_view[i] < r_view[i]) {
                broadcast_dims.emplace_back(i);
                broadcast_shape.m_view[i] = r_view[i];
                broadcast_shape.m_stride[i] = 0;
            }
        }

        return {broadcast_shape, broadcast_dims};
    }

    Shape Shape::reshape(const ShapeView &view) const {
        // TODO: fix this
        is_view_valid(view);
        usize l_numel = get_numel();
        usize r_numel = std::accumulate(view.begin(), view.end(), uone, std::multiplies<usize>());

        if (l_numel != r_numel) {
            throw std::invalid_argument(std::format("cannot reshape array of {} elements to {} elements.", l_numel, r_numel));
        }

        return Shape(m_offset, view);
    }

    ShapeDims Shape::transpose(usize start_dim, usize end_dim) const {
        are_start_end_dim_valid(start_dim, end_dim);
        ShapeDims transpose_dims(get_ndim());
        std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
        std::reverse(transpose_dims.begin() + start_dim, transpose_dims.begin() + end_dim + 1);
        return transpose_dims;
    }

    ShapeView Shape::flatten(usize start_dim, usize end_dim) const {
        are_start_end_dim_valid(start_dim, end_dim);
        ShapeView flatten_view = m_view;
        usize prod = std::accumulate(flatten_view.begin() + start_dim, flatten_view.begin() + end_dim + 1, uone, std::multiplies<usize>());
        // Erase from start_dim + 1 to end_dim + 1
        flatten_view.erase(flatten_view.begin() + start_dim + 1, flatten_view.begin() + end_dim + 1);
        // Update view at start_dim
        flatten_view[start_dim] = prod;
        return flatten_view;
    }

    Shape Shape::permute(const ShapeDims &dims) const {
        is_dim_permutation_valid(dims);
        usize ndim = get_ndim();
        ShapeView view(ndim, 0);
        ShapeStride stride(ndim, 0);

        for (usize i = 0; i < ndim; ++i) {
            view[i] = m_view[dims[i]];
            stride[i] = m_stride[dims[i]];
        }

        return Shape(m_offset, view, stride);
    }

    ShapeDims Shape::undo_permute_dims(const ShapeDims &dims) const {
        is_dim_permutation_valid(dims);
        ShapeDims reverse_dims(dims.size());

        for (std::size_t i = 0; i < dims.size(); ++i) {
            reverse_dims[dims[i]] = i;
        }

        return reverse_dims;
    }

    Shape Shape::undo_permute(const ShapeDims &dims) const {
        return permute(undo_permute_dims(dims));
    }

    Shape Shape::slice(const RangeVec &ranges) const {
        are_ranges_valid(ranges);
        usize offset = m_offset;

        for (std::size_t i = 0; i < ranges.size(); ++i) {
            offset += ranges[i].start() * m_stride[i];
        }

        usize ndim = get_ndim();
        ShapeView view(ndim);
        ShapeStride stride(ndim);

        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const Range &range = ranges[i];
            usize diff = range.stop() - range.start();
            view[i] = static_cast<usize>(ceil((static_cast<double>(diff)) / std::abs(range.step())));
            stride[i] = m_stride[i] * range.step();
        }

        return Shape(offset, view, stride);
    }

    Shape Shape::unsqueeze(const ShapeDims &dims) const {
        ShapeView view = m_view;
        ShapeStride stride = m_stride;

        if (dims.empty()) {
            throw std::invalid_argument("unsqueeze expects at least 1 dimension.");
        }

        usize ndim = get_ndim();
        std::vector<bool> flags(ndim + 1, false);

        for (auto dim : dims) {
            if (dim > ndim) {
                throw std::invalid_argument(std::format("dimension {} is out of range [0, {}] during unsqueeze.", dim, ndim));
            }

            flags[dim] = true;
        }

        for (isize i = ndim; i >= 0; --i) {
            if (flags[i]) {
                view.insert(view.begin() + i, 1);
                stride.insert(stride.begin() + i, 0);
            }
        }

        return Shape(m_offset, view, stride);
    }

    Shape Shape::squeeze(const ShapeDims &dims) const {
        ShapeView view = m_view;
        ShapeStride stride = m_stride;

        if (dims.empty()) {
            for (std::ptrdiff_t i = m_view.size() - 1; i >= 0; --i) {
                if (m_view[i] == 1) {
                    view.erase(view.begin() + i);
                    stride.erase(stride.begin() + i);
                }
            }

            return Shape(m_offset, view, stride);
        }

        usize ndim = get_ndim();
        std::vector<bool> flags(ndim, false);

        for (auto dim : dims) {
            if (dim >= ndim) {
                throw std::invalid_argument(std::format("dimension {} is out of range [0, {}) during squeeze.", dim, ndim));
            }

            if (m_view[dim] != 1) {
                throw std::invalid_argument(std::format("dimension {} is not 1 during squeeze.", dim));
            }

            flags[dim] = true;
        }

        for (isize i = ndim - 1; i >= 0; --i) {
            if (flags[i]) {
                view.erase(view.begin() + i);
                stride.erase(stride.begin() + i);
            }
        }

        return Shape(m_offset, view, stride);
    }
} // namespace nx::foundation
