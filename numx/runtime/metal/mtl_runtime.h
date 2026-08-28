#pragma once

#include "../runtime.h"
#include "mtl_runner.h"

namespace nx::runtime::metal {
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
        static constexpr usize s_full_buffer_count = 2;
        static constexpr usize s_arange_buffer_count = 3;
        static constexpr usize s_uniform_buffer_count = 5;
        static constexpr usize s_contiguous_binary_buffer_count = 4;
        static constexpr usize s_strided_binary_buffer_count = 10;
        static constexpr usize s_contiguous_unary_buffer_count = 3;
        static constexpr usize s_strided_unary_buffer_count = 8;
        static constexpr usize s_contiguous_copy_buffer_count = 3;
        static constexpr usize s_strided_copy_buffer_count = 8;
        static constexpr usize s_contiguous_simd_gevv_buffer_count = 6;
        static constexpr usize s_strided_simd_gevv_buffer_count = 8;
        static constexpr usize s_contiguous_gemm2d_buffer_count = 6;
        static constexpr usize s_strided_gemm2d_buffer_count = 8;
        static constexpr usize s_tensor_gemm2d_buffer_count = 8;
        static constexpr usize s_tensor_gemm2d_tile_size = 32;
        static constexpr usize s_tensor_gemm2d_simdgroups = 2;
        static constexpr usize s_contiguous_gemm3d_buffer_count = 7;
        static constexpr usize s_strided_gemm3d_buffer_count = 9;
        static constexpr usize s_tensor_gemm3d_buffer_count = 10;
        static constexpr usize s_tensor_gemm3d_tile_size = 32;
        static constexpr usize s_tensor_gemm3d_simdgroups = 2;
        static constexpr usize s_contiguous_reduce_all_buffer_count = 4;
        static constexpr usize s_strided_reduce_all_buffer_count = 7;
        static constexpr usize s_contiguous_reduce_col_buffer_count = 4;
        static constexpr usize s_strided_reduce_col_buffer_count = 7;

        void run_full_kernel(Op *op, usize constant) override;
        void run_arange_kernel(Op *op, usize start, usize step) override;
        void run_uniform_kernel(Op *op, std::uint64_t key, usize low, usize high) override;
        void run_binary_kernel(Op *l_op, Op *r_op, Op *out_op) override;
        void run_contiguous_binary_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_strided_binary_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_gevv_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_simd_gevv_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_int_gemm2d_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_tiled_gemm2d_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_tensor_gemm2d_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_int_gemm3d_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_tiled_gemm3d_kernel(Op *l_op, Op *r_op, Op *out_op);
        void run_tensor_gemm3d_kernel(Op *l_op, Op *r_op, Op *out_op);
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
