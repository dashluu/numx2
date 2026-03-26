#include "mtl_encoder.h"
#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_full_kernel(Op *op, usize constant) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        mtl_usize mtl_constant = constant;
        encoder.encode_mtl_buffer(&mtl_constant, sizeof(mtl_usize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "full_" + descriptor.dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRuntime::run_arange_kernel(Op *op, usize start, usize step) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        mtl_usize mtl_start = start;
        mtl_usize mtl_step = step;
        encoder.encode_mtl_buffer(&mtl_start, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(&mtl_step, sizeof(mtl_usize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "arange_" + descriptor.dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRuntime::run_uniform_kernel(Op *op, std::uint64_t key, usize low, usize high) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        usize numel = descriptor.numel();
        mtl_usize mtl_low = low;
        mtl_usize mtl_high = high;
        mtl_usize mtl_numel = numel;
        encoder.encode_mtl_buffer(&key, sizeof(std::uint64_t));
        encoder.encode_mtl_buffer(&mtl_low, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(&mtl_high, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(&mtl_numel, sizeof(mtl_usize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "uniform_" + descriptor.dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        usize num_quad_words = (descriptor.dtype()->size() * numel + 7) / 8;
        encoder.dispatch_threads(num_quad_words, std::min(num_quad_words, s_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace nx::runtime::metal
