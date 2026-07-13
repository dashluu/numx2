#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_reduce_all_kernel(Op *in_op, Op *out_op) {
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        usize numel = in_descriptor.numel();
        mtl_usize mtl_numel = numel;
        bool strided = !in_descriptor.is_contiguous();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        std::uint32_t buff_count = strided ? s_strided_reduce_all_buffer_count : s_contiguous_reduce_all_buffer_count;
        arg_table_desc->setMaxBufferBindCount(buff_count);
        residency_set_desc->setInitialCapacity(buff_count);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        mtl_usize offset[] = {static_cast<mtl_usize>(in_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        runner.encode_mtl_buffer(&mtl_numel, sizeof(mtl_usize));
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);

        if (strided) {
            mtl_usize ndim = in_descriptor.ndim();
            runner.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
            runner.encode_view(in_descriptor);
            runner.encode_stride(in_descriptor);
        }

        runner.encode_array_buffer(in_descriptor);
        runner.encode_array_buffer(out_descriptor);

        // Configure kernel
        const DType *dtype = in_descriptor.dtype();
        std::string kernel_name = std::format("{}{}_all_{}", strided ? "strided_" : "", out_op->opname(), dtype->str());
        runner.commit(kernel_name);

        // Calculate thread configuration
        // Make sure the number of threads aligned to simd size
        usize aligned_numel = foundation::align_to(numel, s_simd_size);
        auto grid_size = MTL::Size::Make(aligned_numel, 1, 1);
        usize threadgroup_nthread = std::min(aligned_numel, s_max_threadgroup_size);
        auto threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);

        // Dispatch kernel
        runner.dispatch_threads(grid_size, threadgroup_size);
        runner.run();
        pool->release();
    }

    std::pair<usize, usize> MTLRuntime::select_reduce_col_kernel_size(usize nrow, usize ncol) {
        // TODO: find a better scheme to determine kernel selection
        usize col_group_size = std::min(s_simd_size, std::bit_floor(ncol));
        usize row_group_size = std::min(std::bit_floor(nrow), s_simd_size / col_group_size);
        return {row_group_size, col_group_size};
    }

    void MTLRuntime::run_reduce_col_kernel(Op *in_op, Op *out_op) {
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        auto reduce_op = static_cast<graph::ReduceOp *>(out_op);
        const ShapeDims &remaining_dims = reduce_op->remaining_dims();
        const ShapeDims &reduce_dims = reduce_op->reduce_dims();
        const ShapeView &in_view = in_descriptor.view();

        // Move reduction dimensions to the end
        ShapeDims permutation_dims;
        permutation_dims.reserve(remaining_dims.size() + reduce_dims.size());
        permutation_dims.insert(permutation_dims.end(), remaining_dims.begin(), remaining_dims.end());
        permutation_dims.insert(permutation_dims.end(), reduce_dims.begin(), reduce_dims.end());
        // Detach input op so the computational graph is not affected
        OpPtr permutation_op = permute(in_op->detach(), permutation_dims);
        const ArrayDescriptor &permutation_descriptor = permutation_op->descriptor();
        share_buffer(permutation_op.get(), in_op);
        usize nrow = std::accumulate(remaining_dims.begin(), remaining_dims.end(), uone, [&](usize acc, usize dim) { return acc * in_view[dim]; });
        usize ncol = std::accumulate(reduce_dims.begin(), reduce_dims.end(), uone, [&](usize acc, usize dim) { return acc * in_view[dim]; });
        mtl_usize mtl_ncol = ncol;
        bool strided = !permutation_descriptor.is_contiguous();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        std::uint32_t buff_count = strided ? s_strided_reduce_col_buffer_count : s_contiguous_reduce_col_buffer_count;
        arg_table_desc->setMaxBufferBindCount(buff_count);
        residency_set_desc->setInitialCapacity(buff_count);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        mtl_usize offset[] = {static_cast<mtl_usize>(permutation_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        runner.encode_mtl_buffer(&mtl_ncol, sizeof(mtl_usize));
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);

        if (strided) {
            mtl_usize ndim = in_descriptor.ndim();
            runner.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
            runner.encode_view(permutation_descriptor);
            runner.encode_stride(permutation_descriptor);
        }

        runner.encode_array_buffer(permutation_descriptor);
        runner.encode_array_buffer(out_descriptor);

        // Configure kernel
        const DType *dtype = permutation_descriptor.dtype();
        auto [row_group_size, col_group_size] = select_reduce_col_kernel_size(nrow, ncol);
        // std::println("row_groups: {}, col_groups: {}", row_groups, col_groups);
        std::string kernel_name = std::format("{}{}_col_{}x{}_{}", strided ? "strided_" : "", out_op->opname(), row_group_size, col_group_size, dtype->str());
        runner.commit(kernel_name);

        // Calculate thread configuration
        auto grid_size = MTL::Size::Make(foundation::align_to(ncol, s_simd_size), nrow, 1);
        auto threadgroup_size = MTL::Size::Make(col_group_size * s_simd_size, row_group_size, 1);

        // Dispatch kernel
        runner.dispatch_threads(grid_size, threadgroup_size);
        runner.run();
        pool->release();
    }
} // namespace nx::runtime::metal
