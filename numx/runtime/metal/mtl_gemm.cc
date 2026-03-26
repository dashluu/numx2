#include "mtl_runtime.h"

namespace nx::runtime::metal {
    using foundation::ShapeView;

    void MTLRuntime::run_gevv_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op) {
        usize numel = l_op->descriptor().numel();
        OpPtr reshaped_l_op = reshape(l_op->detach(), {1, numel});
        OpPtr reshaped_r_op = reshape(r_op->detach(), {numel, 1});
        share_buffer(reshaped_l_op.get(), l_op);
        share_buffer(reshaped_r_op.get(), r_op);
        run_gemm2d_kernel(encoder, reshaped_l_op.get(), reshaped_r_op.get(), out_op);
    }

    void MTLRuntime::run_simd_gevv_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op) {
        // Reset sum to 0
        run_full_kernel(out_op, 0);
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize offset[] = {static_cast<mtl_usize>(l_descriptor.offset()),
                              static_cast<mtl_usize>(r_descriptor.offset()),
                              static_cast<mtl_usize>(out_descriptor.offset())};
        encoder.encode_mtl_buffer(offset, sizeof(mtl_usize) * 3);
        encoder.encode_view(l_descriptor);
        encoder.encode_view(r_descriptor);
        bool strided = !l_descriptor.is_contiguous() || !r_descriptor.is_contiguous();

        if (strided) {
            encoder.encode_stride(l_descriptor);
            encoder.encode_stride(r_descriptor);
        }

        encoder.encode_array_buffer(l_descriptor);
        encoder.encode_array_buffer(r_descriptor);
        encoder.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("{}_{}", strided ? "strided_simd_gevv" : "simd_gevv", l_descriptor.dtype()->str());
        usize numel = l_descriptor.numel();
        auto grid_size = MTL::Size::Make(foundation::align_to(numel, s_simd_size), 1, 1);
        auto threadgroup_size = MTL::Size::Make(s_max_threadgroup_size, 1, 1);
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
    }

    void MTLRuntime::run_gemm2d_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize offset[] = {static_cast<mtl_usize>(l_descriptor.offset()),
                              static_cast<mtl_usize>(r_descriptor.offset()),
                              static_cast<mtl_usize>(out_descriptor.offset())};
        encoder.encode_mtl_buffer(offset, sizeof(mtl_usize) * 3);
        encoder.encode_view(l_descriptor);
        encoder.encode_view(r_descriptor);
        bool strided = !l_descriptor.is_contiguous() || !r_descriptor.is_contiguous();

        if (strided) {
            encoder.encode_stride(l_descriptor);
            encoder.encode_stride(r_descriptor);
        }

        encoder.encode_array_buffer(l_descriptor);
        encoder.encode_array_buffer(r_descriptor);
        encoder.encode_array_buffer(out_descriptor);
        const ShapeView &l_view = l_descriptor.view();
        const ShapeView &r_view = r_descriptor.view();
        std::string kernel_name;
        usize grid_width, grid_height;

        if (foundation::is_float(l_descriptor.dtype())) {
            // Tiling and faster methods can only be used for floating-point
            kernel_name = std::format("{}_{}", strided ? "strided_tiled_gemm2d" : "tiled_gemm2d", l_descriptor.dtype()->str());
            // Tiling uses 8x4 tiles
            grid_width = (r_view[1] + 3) / 4;
            grid_height = (l_view[0] + 7) / 8;
        } else {
            kernel_name = std::format("{}_{}", strided ? "strided_naive_gemm2d" : "naive_gemm2d", l_descriptor.dtype()->str());
            grid_width = r_view[1];
            grid_height = l_view[0];
        }

        auto grid_size = MTL::Size::Make(grid_width, grid_height, 1);
        auto threadgroup_size = MTL::Size::Make(s_threadgroup_size, 1, 1);
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
    }

    void MTLRuntime::run_gemm3d_kernel(MTLEncoder &encoder, Op *l_op, Op *r_op, Op *out_op) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize ndim = l_descriptor.ndim();
        mtl_usize offset[] = {static_cast<mtl_usize>(l_descriptor.offset()),
                              static_cast<mtl_usize>(r_descriptor.offset()),
                              static_cast<mtl_usize>(out_descriptor.offset())};
        encoder.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(offset, sizeof(mtl_usize) * 3);
        encoder.encode_view(l_descriptor);
        encoder.encode_view(r_descriptor);
        bool strided = !l_descriptor.is_contiguous() || !r_descriptor.is_contiguous();

        if (strided) {
            encoder.encode_stride(l_descriptor);
            encoder.encode_stride(r_descriptor);
        }

        encoder.encode_array_buffer(l_descriptor);
        encoder.encode_array_buffer(r_descriptor);
        encoder.encode_array_buffer(out_descriptor);
        const ShapeView &l_view = l_descriptor.view();
        const ShapeView &r_view = r_descriptor.view();
        usize batch_size = std::accumulate(l_view.begin(), l_view.end() - 2, uone, std::multiplies<usize>());
        std::string kernel_name;
        usize grid_width, grid_height;

        if (foundation::is_float(l_descriptor.dtype())) {
            // Tiling and faster methods can only be used for floating-point
            kernel_name = std::format("{}_{}", strided ? "strided_tiled_gemm3d" : "tiled_gemm3d", l_descriptor.dtype()->str());
            // Tiling uses 8x4 tiles
            grid_width = (r_view[ndim - 1] + 3) / 4;
            grid_height = (l_view[ndim - 2] + 7) / 8;
        } else {
            kernel_name = std::format("{}_{}", strided ? "strided_naive_gemm3d" : "naive_gemm3d", l_descriptor.dtype()->str());
            grid_width = r_view[ndim - 1];
            grid_height = l_view[ndim - 2];
        }

        auto grid_size = MTL::Size::Make(grid_width, grid_height, batch_size);
        auto threadgroup_size = MTL::Size::Make(s_threadgroup_size, 1, 1);
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
    }

    void MTLRuntime::run_gemm_kernel(Op *l_op, Op *r_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());

        switch (l_op->descriptor().ndim()) {
        case 1:
            run_gevv_kernel(encoder, l_op, r_op, out_op);
            // run_simd_gevv_kernel(encoder, l_op, r_op, out_op);
            break;
        case 2:
            run_gemm2d_kernel(encoder, l_op, r_op, out_op);
            break;
        case 3:
            run_gemm3d_kernel(encoder, l_op, r_op, out_op);
            break;
        default:
            run_gemm3d_kernel(encoder, l_op, r_op, out_op);
            break;
        }

        pool->release();
    }
} // namespace nx::runtime::metal
