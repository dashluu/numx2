#pragma once

#include "../runtime.h"
#include "mtl_encoder.h"

namespace nx::runtime::metal {
    using mtl_usize = std::uint32_t;
    using mtl_isize = std::int32_t;
    using foundation::ArrayBufferBorrower;
    using foundation::ArrayBufferOwner;
    using foundation::Buffer;
    using foundation::DType;
    using foundation::ShapeDims;
    using foundation::ShapeView;
    using foundation::uone;
    using graph::Opcode;
    using graph::OpPtr;

    class MTLRuntime : public Runtime {
    private:
        static constexpr usize s_simd_size = 32;
        static constexpr usize s_threadgroup_size = 256;
        static constexpr usize s_max_threadgroup_size = 1024;

        void run_full_kernel(Op *op, usize constant) override;
        void run_arange_kernel(Op *op, isize start, isize step) override;
        void run_binary_kernel(Op *l_op, Op *r_op, Op *out_op) override;
        void run_contiguous_binary_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_strided_binary_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_gevv_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op);
        void run_simd_gevv_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op);
        void run_gemm2d_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op);
        void run_gemm3d_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op);
        void run_gemm_kernel(Op *l_op, Op *r_op, Op *out_op) override;
        void run_unary_kernel(Op *in_op, Op *out_op) override;
        void run_contiguous_unary_kernel(Op *in_op, Op *out_op);
        void run_strided_unary_kernel(Op *in_op, Op *out_op);
        void run_copy_kernel(Op *in_op, Op *out_op) override;
        void run_contiguous_copy_kernel(Op *in_op, Op *out_op);
        void run_strided_copy_kernel(Op *in_op, Op *out_op);
        void run_reduce_all_kernel(Op *in_op, Op *out_op) override;
        std::pair<usize, usize> select_reduce_col_kernel_size(usize nrow, usize ncol);
        void run_reduce_col_kernel(Op *in_op, Op *out_op) override;
        void run_initializer_op(Op *op) override;
        void run_unary_op(Op *op) override;
        void run_binary_op(Op *op) override;
        void run_transform_op(Op *op) override;
        void run_reduce_op(Op *op) override;

        template <class O>
        void run_simple_transform_op(Op *op) {
            auto transform_op = static_cast<O *>(op);
            Op *operand = transform_op->operand_raw();
            share_buffer(op, operand);
        }

        void alloc_buffer(Op *op);
        void share_buffer(Op *dest, Op *src);

    public:
        explicit MTLRuntime(RuntimeContextPtr ctx) : Runtime(std::move(ctx)) {}
    };
} // namespace nx::runtime::metal
