#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_contiguous_unary_kernel(Op *in_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        const mtl_usize offset[] = {static_cast<mtl_usize>(in_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        encoder.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);
        encoder.encode_array_buffer(in_descriptor);
        encoder.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("{}_{}", out_op->opname(), in_descriptor.dtype()->str());
        encoder.set_pipeline_state(kernel_name);
        usize numel = in_descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRuntime::run_strided_unary_kernel(Op *in_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize ndim = in_descriptor.ndim();
        const mtl_usize offset[] = {static_cast<mtl_usize>(in_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        const bool strided[] = {!in_descriptor.is_contiguous(), !out_descriptor.is_contiguous()};
        encoder.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);
        encoder.encode_view(in_descriptor);
        encoder.encode_stride(in_descriptor);
        encoder.encode_stride(out_descriptor);
        encoder.encode_mtl_buffer(strided, sizeof(bool) * 2);
        encoder.encode_array_buffer(in_descriptor);
        encoder.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("strided_{}_{}", out_op->opname(), in_descriptor.dtype()->str());
        encoder.set_pipeline_state(kernel_name);
        usize numel = in_descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRuntime::run_unary_kernel(Op *in_op, Op *out_op) {
        if (in_op->descriptor().is_contiguous() && out_op->descriptor().is_contiguous()) {
            run_contiguous_unary_kernel(in_op, out_op);
        } else {
            run_strided_unary_kernel(in_op, out_op);
        }
    }
} // namespace nx::runtime::metal
