#pragma once

#include "../graph/graph.h"
#include "runtime_context.h"

namespace nx::runtime {
    using foundation::isize;
    using foundation::usize;
    using graph::Graph;
    using graph::Op;
    using graph::PrimitiveKind;

    class Runtime {
    protected:
        RuntimeContextPtr m_ctx;

        virtual void run_full_kernel(Op *op, usize constant) = 0;
        virtual void run_arange_kernel(Op *op, isize start, isize step) = 0;
        virtual void run_binary_kernel(Op *l_op, Op *r_op, Op *out_op) = 0;
        virtual void run_gemm_kernel(Op *l_op, Op *r_op, Op *out_op) = 0;
        virtual void run_unary_kernel(Op *in_op, Op *out_op) = 0;
        virtual void run_copy_kernel(Op *in_op, Op *out_op) = 0;
        virtual void run_reduce_all_kernel(Op *in_op, Op *out_op) = 0;
        virtual void run_reduce_col_kernel(Op *in_op, Op *out_op) = 0;
        virtual void run_initializer_op(Op *op) = 0;
        virtual void run_unary_op(Op *op) = 0;
        virtual void run_binary_op(Op *op) = 0;
        virtual void run_transform_op(Op *op) = 0;
        virtual void run_reduce_op(Op *op) = 0;
        void run_op(Op *op);

    public:
        explicit Runtime(RuntimeContextPtr ctx) : m_ctx(std::move(ctx)) {}
        Runtime(const Runtime &) = delete;
        Runtime(Runtime &&) noexcept = delete;
        virtual ~Runtime() = default;
        Runtime &operator=(const Runtime &) = delete;
        Runtime &operator=(Runtime &&) noexcept = delete;
        const RuntimeContext *context() const { return m_ctx.get(); }
        void forward(Graph *graph);
        void backward(Graph *graph);
    };

    using RuntimePtr = std::unique_ptr<Runtime>;

    template <class T>
    concept DerivedRuntime = std::is_base_of<Runtime, T>::value;

    template <DerivedRuntime T, class... Args>
    std::unique_ptr<T> make_runtime(Args &&...args) { return std::make_unique<T>(std::forward<Args>(args)...); }
} // namespace nx::runtime
