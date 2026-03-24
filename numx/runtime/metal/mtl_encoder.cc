#include "mtl_encoder.h"
#include "mtl_runtime.h"

namespace nx::runtime::metal {
    MTLEncoder::~MTLEncoder() {
        for (auto &buff : m_mtl_buffs) {
            buff->release();
        }
    }

    void MTLEncoder::encode_mtl_buffer(const void *buff, usize size) {
        MTL::Buffer *mtl_buff = m_ctx->mtl_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
        m_mtl_buffs.push_back(mtl_buff);
        m_encoder->setBuffer(mtl_buff, 0, m_buff_idx);
        ++m_buff_idx;
    }

    void MTLEncoder::encode_view(const ArrayDescriptor &descriptor) {
        const ShapeView &view = descriptor.view();
        usize ndim = descriptor.ndim();

        for (usize i = 0; i < ndim; ++i) {
            m_cpu_buff.push_back(view[i]);
        }

        usize *shape_buff = m_cpu_buff.data() + (m_cpu_buff.size() - ndim);
        encode_mtl_buffer(shape_buff, ndim);
    }

    void MTLEncoder::encode_stride(const ArrayDescriptor &descriptor) {
        const ShapeStride &stride = descriptor.stride();
        usize ndim = descriptor.ndim();

        for (usize i = 0; i < ndim; ++i) {
            m_cpu_buff.push_back(stride[i]);
        }

        usize *stride_buff = m_cpu_buff.data() + (m_cpu_buff.size() - ndim);
        encode_mtl_buffer(stride_buff, ndim);
    }

    void MTLEncoder::set_pipeline_state(const std::string &kernel_name) {
        MTLKernel *kernel = m_ctx->kernel(kernel_name);

        if (!kernel) {
            throw std::runtime_error(std::format("No kernel named {}.", kernel_name));
        }

        m_encoder->setComputePipelineState(kernel->state().get());
    }

    void MTLEncoder::dispatch_threads(usize grid_nthread, usize threadgroup_nthread) {
        MTL::Size grid_size = MTL::Size::Make(grid_nthread, 1, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);
        dispatch_threads(grid_size, threadgroup_size);
    }

    void MTLEncoder::dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size) {
        m_encoder->dispatchThreads(grid_size, threadgroup_size);
        m_encoder->endEncoding();
        m_cmd_buff->commit();
    }

    double MTLEncoder::time_to_complete() {
        m_cmd_buff->waitUntilCompleted();
        CFTimeInterval start = m_cmd_buff->GPUStartTime();
        CFTimeInterval end = m_cmd_buff->GPUEndTime();
        return end - start;
    }
} // namespace nx::runtime::metal
