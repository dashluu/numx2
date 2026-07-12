#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_contiguous_binary_kernel(Op *l_op, Op *r_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(4);
        residency_set_desc->setInitialCapacity(4);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize offset[] = {static_cast<mtl_usize>(l_descriptor.offset()),
                              static_cast<mtl_usize>(r_descriptor.offset()),
                              static_cast<mtl_usize>(out_descriptor.offset())};
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 3);
        runner.encode_array_buffer(l_descriptor);
        runner.encode_array_buffer(r_descriptor);
        runner.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("{}_{}", out_op->opname(), l_descriptor.dtype()->str());
        runner.commit(kernel_name);
        usize numel = l_descriptor.numel();
        runner.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        runner.run();
        pool->release();
    }

    void MTLRuntime::run_strided_binary_kernel(Op *l_op, Op *r_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(10);
        residency_set_desc->setInitialCapacity(10);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize ndim = l_descriptor.ndim();
        mtl_usize offset[] = {static_cast<mtl_usize>(l_descriptor.offset()),
                              static_cast<mtl_usize>(r_descriptor.offset()),
                              static_cast<mtl_usize>(out_descriptor.offset())};
        bool strided[] = {!l_descriptor.is_contiguous(), !r_descriptor.is_contiguous(), !out_descriptor.is_contiguous()};
        runner.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 3);
        runner.encode_view(l_descriptor);
        runner.encode_stride(l_descriptor);
        runner.encode_stride(r_descriptor);
        runner.encode_stride(out_descriptor);
        runner.encode_mtl_buffer(strided, sizeof(bool) * 3);
        runner.encode_array_buffer(l_descriptor);
        runner.encode_array_buffer(r_descriptor);
        runner.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("strided_{}_{}", out_op->opname(), l_descriptor.dtype()->str());
        runner.commit(kernel_name);
        usize numel = l_descriptor.numel();
        runner.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        runner.run();
        pool->release();
    }

    void MTLRuntime::run_binary_kernel(Op *l_op, Op *r_op, Op *out_op) {
        if (l_op->descriptor().is_contiguous() &&
            r_op->descriptor().is_contiguous() &&
            out_op->descriptor().is_contiguous()) {
            run_contiguous_binary_kernel(l_op, r_op, out_op);
        } else {
            run_strided_binary_kernel(l_op, r_op, out_op);
        }
    }
} // namespace nx::runtime::metal
