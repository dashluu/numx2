#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_full_kernel(Op *op, usize constant) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        const DType *dtype = descriptor.dtype();
        encoder.encode_mtl_buffer(&constant, dtype->size());
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "full_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRuntime::run_arange_kernel(Op *op, isize start, isize step) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        mtl_isize mtl_start = start;
        mtl_isize mtl_step = step;
        encoder.encode_mtl_buffer(&mtl_start, sizeof(mtl_isize));
        encoder.encode_mtl_buffer(&mtl_step, sizeof(mtl_isize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "arange_" + descriptor.dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace nx::runtime::metal
