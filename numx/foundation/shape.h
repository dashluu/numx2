#pragma once

#include "range.h"
#include "utils.h"
#include <algorithm>
#include <initializer_list>
#include <numeric>

namespace nx::foundation {
    using InitList = std::initializer_list<usize>;
    using ShapeView = std::vector<usize>;
    using ShapeStride = std::vector<isize>;
    using ShapeDims = std::vector<usize>;

    class Shape {
    private:
        using RangeVec = std::vector<Range>;
        usize m_offset;
        ShapeView m_view;
        ShapeStride m_stride;

        void are_ranges_valid(const RangeVec &ranges) const;
        void is_dim_permutation_valid(const ShapeDims &dims) const;
        void are_start_end_dim_valid(usize start_dim, usize end_dim) const;
        static void is_view_valid(const ShapeView &view);

    public:
        Shape() : Shape(0, {1}, {1}) {}
        Shape(usize offset, const ShapeView &view, const ShapeStride &stride);
        Shape(usize offset, const ShapeView &view);
        Shape(InitList view) : Shape(0, view) {}
        explicit Shape(ShapeView view) : Shape(0, std::move(view)) {}
        Shape(const Shape &) = default;
        Shape(Shape &&) noexcept = default;
        Shape &operator=(const Shape &) = default;
        Shape &operator=(Shape &&) noexcept = default;
        usize get_offset() const { return m_offset; }
        const ShapeView &get_view() const { return m_view; }
        const ShapeStride &get_stride() const { return m_stride; }
        bool operator==(const Shape &shape) const { return m_view == shape.m_view; }
        usize operator[](usize dim) const { return m_view[dim]; }
        ShapeView::const_iterator begin() const { return m_view.cbegin(); }
        ShapeView::const_iterator end() const { return m_view.cend(); }
        std::string str() const { return std::format("offset: {}, view: ({}), stride: ({})", m_offset, join_nums(m_view), join_nums(m_stride)); }
        friend std::ostream &operator<<(std::ostream &os, const Shape &shape) { return os << shape.str(); }
        usize get_ndim() const { return m_view.size(); }
        usize get_numel() const { return std::accumulate(m_view.begin(), m_view.end(), uone, std::multiplies<usize>()); }
        bool is_contiguous() const { return m_stride == contiguous_stride(); }
        ShapeStride contiguous_stride() const;
        std::vector<usize> size_per_dim() const;
        usize get_size(usize dim) const;
        bool broadcastable(const ShapeView &view) const;
        // One-direction broadcast check
        bool broadcastable_to(const ShapeView &view) const;
        bool gemm_broadcastable(const ShapeView &view) const;
        // One-direction broadcast
        std::pair<Shape, ShapeDims> broadcast_to(const ShapeView &view) const;
        // ShapeDims specifies which dimensions are broadcasted
        std::pair<Shape, ShapeDims> broadcast(const ShapeView &view) const;
        Shape reshape(const ShapeView &view) const;
        ShapeDims transpose(usize start_dim, usize end_dim) const;
        ShapeView flatten(usize start_dim, usize end_dim) const;
        Shape permute(const ShapeDims &dims) const;
        ShapeDims undo_permute_dims(const ShapeDims &dims) const;
        Shape undo_permute(const ShapeDims &dims) const;
        Shape slice(const RangeVec &ranges) const;
        Shape unsqueeze(const ShapeDims &dims) const;
        Shape squeeze(const ShapeDims &dims) const;
    };
} // namespace nx::foundation
