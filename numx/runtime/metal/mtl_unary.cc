#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_contiguous_unary_kernel(Op *in_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(3);
        residency_set_desc->setInitialCapacity(3);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize offset[] = {static_cast<mtl_usize>(in_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);
        runner.encode_array_buffer(in_descriptor);
        runner.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("{}_{}", out_op->opname(), in_descriptor.dtype()->str());
        runner.commit(kernel_name);
        usize numel = in_descriptor.numel();
        runner.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        runner.run();
        pool->release();
    }

    void MTLRuntime::run_strided_unary_kernel(Op *in_op, Op *out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::SharedPtr<MTL4::ArgumentTableDescriptor> arg_table_desc = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        NS::SharedPtr<MTL::ResidencySetDescriptor> residency_set_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        arg_table_desc->setMaxBufferBindCount(8);
        residency_set_desc->setInitialCapacity(8);
        MTLRunner runner(m_ctx.get(), arg_table_desc.get(), residency_set_desc.get());
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ArrayDescriptor &out_descriptor = out_op->descriptor();
        mtl_usize ndim = in_descriptor.ndim();
        mtl_usize offset[] = {static_cast<mtl_usize>(in_descriptor.offset()), static_cast<mtl_usize>(out_descriptor.offset())};
        bool strided[] = {!in_descriptor.is_contiguous(), !out_descriptor.is_contiguous()};
        runner.encode_mtl_buffer(&ndim, sizeof(mtl_usize));
        runner.encode_mtl_buffer(offset, sizeof(mtl_usize) * 2);
        runner.encode_view(in_descriptor);
        runner.encode_stride(in_descriptor);
        runner.encode_stride(out_descriptor);
        runner.encode_mtl_buffer(strided, sizeof(bool) * 2);
        runner.encode_array_buffer(in_descriptor);
        runner.encode_array_buffer(out_descriptor);
        std::string kernel_name = std::format("strided_{}_{}", out_op->opname(), in_descriptor.dtype()->str());
        runner.commit(kernel_name);
        usize numel = in_descriptor.numel();
        runner.dispatch_threads(numel, std::min(numel, s_threadgroup_size));
        runner.run();
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
