#include "mtl_runtime.h"

namespace nx::runtime::metal {
    MTLRunner::MTLRunner(RuntimeContext *ctx,
                         MTL4::ArgumentTableDescriptor *arg_table_desc,
                         MTL::ResidencySetDescriptor *residency_set_desc) {
        m_ctx = static_cast<MTLContext *>(ctx);
        auto device = m_ctx->mtl_device();
        NS::Error *error = nullptr;
        m_arg_table = NS::TransferPtr<MTL4::ArgumentTable>(device->newArgumentTable(arg_table_desc, &error));

        if (error) {
            const std::string description = error->localizedDescription()->utf8String();
            throw std::runtime_error(description);
        }

        m_residency_set = NS::TransferPtr<MTL::ResidencySet>(device->newResidencySet(residency_set_desc, &error));

        if (error) {
            const std::string description = error->localizedDescription()->utf8String();
            throw std::runtime_error(description);
        }

        auto cmd_buff = m_ctx->cmd_buff();
        auto cmd_allocator = m_ctx->cmd_allocator();
        cmd_allocator->reset();
        cmd_buff->beginCommandBuffer(cmd_allocator.get());
        m_encoder = cmd_buff->computeCommandEncoder();
        m_event = NS::TransferPtr<MTL::SharedEvent>(device->newSharedEvent());
    }

    MTLRunner::~MTLRunner() {
        for (auto &buff : m_encoded_buffs) {
            delete[] buff;
        }

        for (auto &buff : m_mtl_buffs) {
            buff->release();
        }
    }

    void MTLRunner::encode_mtl_buffer(const void *buff, usize size) {
        MTL::Buffer *mtl_buff = m_ctx->mtl_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
        m_mtl_buffs.emplace_back(mtl_buff);
        m_residency_set->addAllocation(mtl_buff);
        m_arg_table->setAddress(mtl_buff->gpuAddress(), m_buff_idx);
        ++m_buff_idx;
    }

    void MTLRunner::encode_view(const ArrayDescriptor &descriptor) {
        const ShapeView &view = descriptor.view();
        usize ndim = descriptor.ndim();
        mtl_usize *view_buff = new mtl_usize[ndim];

        for (usize i = 0; i < ndim; ++i) {
            view_buff[i] = view[i];
        }

        m_encoded_buffs.emplace_back(view_buff);
        encode_mtl_buffer(view_buff, ndim * sizeof(mtl_usize));
    }

    void MTLRunner::encode_stride(const ArrayDescriptor &descriptor) {
        const ShapeStride &stride = descriptor.stride();
        usize ndim = descriptor.ndim();
        mtl_usize *stride_buff = new mtl_usize[ndim];

        for (usize i = 0; i < ndim; ++i) {
            stride_buff[i] = stride[i];
        }

        m_encoded_buffs.emplace_back(stride_buff);
        encode_mtl_buffer(stride_buff, ndim * sizeof(mtl_usize));
    }

    void MTLRunner::commit(const std::string &kernel_name) {
        MTLKernel *kernel = m_ctx->kernel(kernel_name);

        if (!kernel) {
            throw std::runtime_error(std::format("no kernel named {}.", kernel_name));
        }

        auto cmd_buff = m_ctx->cmd_buff();
        m_residency_set->commit();
        cmd_buff->useResidencySet(m_residency_set.get());
        m_encoder->setComputePipelineState(kernel->state().get());
        m_encoder->setArgumentTable(m_arg_table.get());
    }

    void MTLRunner::dispatch_threads(usize grid_nthread, usize threadgroup_nthread) {
        MTL::Size grid_size = MTL::Size::Make(grid_nthread, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);
        dispatch_threads(grid_size, threadgroup_size);
    }

    void MTLRunner::dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size) {
        m_encoder->dispatchThreads(grid_size, threadgroup_size);
    }

    void MTLRunner::run() {
        auto cmd_buff = m_ctx->cmd_buff();
        m_encoder->endEncoding();
        cmd_buff->endCommandBuffer();
        const MTL4::CommandBuffer *cmd_buffs[] = {cmd_buff.get()};
        auto cmd_queue = m_ctx->cmd_queue();
        cmd_queue->commit(cmd_buffs, 1);
        std::uint64_t signal = 1;
        cmd_queue->signalEvent(m_event.get(), signal);
        m_event->waitUntilSignaledValue(signal, 1000);
    }
} // namespace nx::runtime::metal
