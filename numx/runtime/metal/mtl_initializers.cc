#include "mtl_encoder.h"
#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_full_kernel(Op *op, usize constant) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(2);
        residency_set_desc->setInitialCapacity(2);
        MTLEncoder encoder(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        mtl_usize mtl_constant = constant;
        encoder.encode_mtl_buffer(&mtl_constant, sizeof(mtl_usize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "full_" + descriptor.dtype()->str();
        encoder.use_kernel(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.commit();
        pool->release();
    }

    void MTLRuntime::run_arange_kernel(Op *op, usize start, usize step) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(3);
        residency_set_desc->setInitialCapacity(3);
        MTLEncoder encoder(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &descriptor = op->descriptor();
        mtl_usize mtl_start = start;
        mtl_usize mtl_step = step;
        encoder.encode_mtl_buffer(&mtl_start, sizeof(mtl_usize));
        encoder.encode_mtl_buffer(&mtl_step, sizeof(mtl_usize));
        encoder.encode_array_buffer(descriptor);
        std::string kernel_name = "arange_" + descriptor.dtype()->str();
        encoder.use_kernel(kernel_name);
        usize numel = descriptor.numel();
        encoder.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        encoder.commit();
        pool->release();
    }

    void MTLRuntime::run_uniform_kernel(Op *op, std::uint64_t key, usize low, usize high) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(5);
        residency_set_desc->setInitialCapacity(5);
        MTLEncoder encoder(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
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
        encoder.use_kernel(kernel_name);
        usize num_quad_words = (descriptor.dtype()->size() * numel + 7) / 8;
        encoder.dispatch_threads(num_quad_words, std::min(num_quad_words, s_threadgroup_size));
        encoder.commit();
        pool->release();
    }
} // namespace nx::runtime::metal
