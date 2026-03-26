#pragma once

#include "../graph/op.h"
#include "backend.h"

namespace nx::core {
    using foundation::ArrayBuffer;
    using foundation::ArrayDescriptor;
    using foundation::default_device_name;
    using foundation::Device;
    using foundation::DType;
    using foundation::f32;
    using foundation::isize;
    using foundation::NumericOrBoolType;
    using foundation::NumericType;
    using foundation::RangeVec;
    using foundation::Shape;
    using foundation::ShapeDims;
    using foundation::ShapeStride;
    using foundation::ShapeView;
    using foundation::usize;
    using graph::GraphPtr;
    using graph::Op;
    using graph::OpPtr;
    using memory::BufferMemory;
    using runtime::Runtime;
    using runtime::RuntimeContext;

    Runtime *backend_runtime(usize device_id, DeviceKind device_kind);

    struct Array {
    protected:
        OpPtr m_op = nullptr;
        GraphPtr m_graph = nullptr;
        Runtime *m_runtime = nullptr;

    public:
        Array() = default;
        Array(OpPtr op);
        Array(OpPtr op, Runtime *runtime) : m_op(op), m_runtime(runtime) {}
        Array(const Array &) = default;
        Array(Array &&) noexcept = default;
        virtual ~Array();
        Array &operator=(const Array &) = default;
        Array &operator=(Array &&) noexcept = default;
        OpPtr op() const { return m_op; }
        GraphPtr graph() const { return m_graph; }
        const ArrayDescriptor &descriptor() const { return m_op->descriptor(); }
        const Shape &shape() const { return descriptor().shape(); }
        usize offset() const { return descriptor().offset(); }
        const ShapeView &view() const { return descriptor().view(); }
        const ShapeStride &stride() const { return descriptor().stride(); }
        std::uint8_t *ptr() const { return m_op->descriptor().ptr(); }
        const DType *dtype() const { return descriptor().dtype(); }
        const Device *device() const { return descriptor().device(); }

        std::optional<Array> grad() const {
            OpPtr grad = m_op->grad();
            return grad ? std::optional<Array>(Array(graph::detach(grad))) : std::nullopt;
        }

        usize numel() const { return descriptor().numel(); }
        usize ndim() const { return descriptor().ndim(); }
        usize itemsize() const { return descriptor().itemsize(); }
        usize nbytes() const { return descriptor().nbytes(); }
        usize size(usize dim) const { return descriptor().size(dim); }
        bool is_grad_enabled() const { return m_op->is_grad_enabled(); }
        // This can only be used before compilation or forwarding, otherwise, there is no effect
        void enable_grad(bool enabled) { m_op->enable_grad(enabled); }
        bool is_parameter() const { return m_op->is_param(); }
        bool is_contiguous() const { return descriptor().is_contiguous(); }

        usize item() {
            eval();
            return graph::item(m_op);
        }

        // std::string graph_str() {
        //     eval();
        //     return m_graph->str();
        // }

        std::string str() {
            eval();
            return m_op->descriptor().str();
        }

        void eval();
        void backward();
        friend std::ostream &operator<<(std::ostream &os, Array &array) { return os << array.str(); }
        Array detach(bool is_param = false) const { return Array(graph::detach(m_op, is_param)); }

        // Element-wise operations
        Array operator+(const Array &rhs) const { return Array(graph::add(m_op, rhs.m_op)); }

        template <NumericType T>
        Array operator+(T constant) const { return Array(graph::add(m_op, constant)); }

        Array operator-(const Array &rhs) const { return Array(graph::sub(m_op, rhs.m_op)); }

        template <NumericType T>
        Array operator-(T constant) const { return Array(graph::sub(m_op, constant)); }

        Array operator*(const Array &rhs) const { return Array(graph::mul(m_op, rhs.m_op)); }

        template <NumericType T>
        Array operator*(T constant) const { return Array(graph::mul(m_op, constant)); }

        Array operator/(const Array &rhs) const { return Array(graph::div(m_op, rhs.m_op)); }

        template <NumericType T>
        Array operator/(T constant) const { return Array(graph::div(m_op, constant)); }

        Array &operator+=(const Array &rhs) {
            m_op = graph::i_add(m_op, rhs.m_op);
            m_graph = nullptr;
            return *this;
        }

        template <NumericType T>
        Array &operator+=(T constant) {
            m_op = graph::i_add(m_op, constant);
            m_graph = nullptr;
            return *this;
        }

        Array &operator-=(const Array &rhs) {
            m_op = graph::i_sub(m_op, rhs.m_op);
            m_graph = nullptr;
            return *this;
        }

        template <NumericType T>
        Array &operator-=(T constant) {
            m_op = graph::i_sub(m_op, constant);
            m_graph = nullptr;
            return *this;
        }

        Array &operator*=(const Array &rhs) {
            m_op = graph::i_mul(m_op, rhs.m_op);
            m_graph = nullptr;
            return *this;
        }

        template <NumericType T>
        Array &operator*=(T constant) {
            m_op = graph::i_mul(m_op, constant);
            m_graph = nullptr;
            return *this;
        }

        Array &operator/=(const Array &rhs) {
            m_op = graph::i_div(m_op, rhs.m_op);
            m_graph = nullptr;
            return *this;
        }

        template <NumericType T>
        Array &operator/=(T constant) {
            m_op = graph::i_div(m_op, constant);
            m_graph = nullptr;
            return *this;
        }

        Array gemm(const Array &rhs) const { return Array(graph::gemm(m_op, rhs.m_op)); }
        Array exp(bool in_place = false) const { return Array(graph::exp(m_op, in_place)); }
        Array log(bool in_place = false) const { return Array(graph::log(m_op, in_place)); }
        Array sqrt(bool in_place = false) const { return Array(graph::sqrt(m_op, in_place)); }
        Array sq(bool in_place = false) const { return Array(graph::sq(m_op, in_place)); }
        Array neg(bool in_place = false) const { return Array(graph::neg(m_op, in_place)); }
        Array operator-() const { return Array(graph::neg(m_op)); }
        Array recip(bool in_place = false) const { return Array(graph::recip(m_op, in_place)); }
        Array sin(bool in_place = false) const { return Array(graph::sin(m_op, in_place)); }
        Array cos(bool in_place = false) const { return Array(graph::cos(m_op, in_place)); }
        Array operator==(const Array &rhs) const { return Array(graph::eq(m_op, rhs.m_op)); }
        Array operator!=(const Array &rhs) const { return Array(graph::neq(m_op, rhs.m_op)); }
        Array operator<(const Array &rhs) const { return Array(graph::less(m_op, rhs.m_op)); }
        Array operator>(const Array &rhs) const { return Array(graph::greater(m_op, rhs.m_op)); }
        Array operator<=(const Array &rhs) const { return Array(graph::leq(m_op, rhs.m_op)); }
        Array operator>=(const Array &rhs) const { return Array(graph::geq(m_op, rhs.m_op)); }
        Array minimum(const Array &rhs) const { return Array(graph::minimum(m_op, rhs.m_op)); }
        Array maximum(const Array &rhs) const { return Array(graph::maximum(m_op, rhs.m_op)); }

        template <NumericOrBoolType T>
        Array operator==(T constant) const { return Array(graph::eq(m_op, constant)); }

        template <NumericOrBoolType T>
        Array operator!=(T constant) const { return Array(graph::neq(m_op, constant)); }

        template <NumericType T>
        Array operator<(T constant) const { return Array(graph::less(m_op, constant)); }

        template <NumericType T>
        Array operator>(T constant) const { return Array(graph::greater(m_op, constant)); }

        template <NumericType T>
        Array operator<=(T constant) const { return Array(graph::leq(m_op, constant)); }

        template <NumericType T>
        Array operator>=(T constant) const { return Array(graph::geq(m_op, constant)); }

        template <NumericType T>
        Array minimum(T constant) const { return Array(graph::minimum(m_op, constant)); }

        template <NumericType T>
        Array maximum(T constant) const { return Array(graph::maximum(m_op, constant)); }

        // Reduction operations
        Array sum(const ShapeDims &dims = {}) const { return Array(graph::sum(m_op, dims)); }
        Array mean(const ShapeDims &dims = {}) const { return Array(graph::mean(m_op, dims)); }
        Array max(const ShapeDims &dims = {}) const { return Array(graph::max(m_op, dims)); }
        Array min(const ShapeDims &dims = {}) const { return Array(graph::min(m_op, dims)); }
        Array argmax(const ShapeDims &dims = {}) const { return Array(graph::argmax(m_op, dims)); }
        Array argmin(const ShapeDims &dims = {}) const { return Array(graph::argmin(m_op, dims)); }

        // Shape operations
        Array broadcast(const ShapeView &view) const { return Array(graph::broadcast(m_op, view)); }
        Array broadcast_to(const ShapeView &view) const { return Array(graph::broadcast_to(m_op, view)); }
        Array slice(const RangeVec &ranges) const { return Array(graph::slice(m_op, ranges)); }
        Array reshape(const ShapeView &view) const { return Array(graph::reshape(m_op, view)); }
        Array flatten(usize start_dim, usize end_dim) const { return Array(graph::flatten(m_op, start_dim, end_dim)); }
        Array squeeze(const ShapeDims &dims = {}) const { return Array(graph::squeeze(m_op, dims)); }
        Array unsqueeze(const ShapeDims &dims = {}) const { return Array(graph::unsqueeze(m_op, dims)); }
        Array permute(const ShapeDims &dims) const { return Array(graph::permute(m_op, dims)); }
        Array transpose(usize start_dim, usize end_dim) const { return Array(graph::transpose(m_op, start_dim, end_dim)); }

        // Type operations
        Array astype(const DType *dtype) const { return Array(graph::astype(m_op, dtype)); }
    };

    using ArrayVec = std::vector<Array>;
} // namespace nx::core

namespace std {
    template <>
    struct formatter<nx::core::Array> : formatter<string> {
        auto format(nx::core::Array &array, format_context &ctx) const {
            return formatter<string>::format(array.str(), ctx);
        }
    };
} // namespace std
