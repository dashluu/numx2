#pragma once

#include "mtl_context.h"

namespace nx::runtime::metal {
    using foundation::ArrayBuffer;
    using foundation::ArrayDescriptor;
    using foundation::ShapeStride;
    using foundation::ShapeView;
    using memory::Buffer;

    class MTLEncoder {
    private:
        MTLContext *m_ctx;
        MTL::CommandBuffer *m_cmd_buff;
        MTL::ComputeCommandEncoder *m_encoder;
        std::vector<usize> m_cpu_buff;
        std::vector<MTL::Buffer *> m_mtl_buffs;
        usize m_buff_idx = 0;

    public:
        explicit MTLEncoder(RuntimeContext *ctx) {
            m_ctx = static_cast<MTLContext *>(ctx);
            m_cmd_buff = m_ctx->cmd_queue()->commandBuffer();
            m_encoder = m_cmd_buff->computeCommandEncoder();
        }

        MTLEncoder(const MTLEncoder &) = delete;
        MTLEncoder(MTLEncoder &&) noexcept = delete;
        ~MTLEncoder();

        MTLEncoder &operator=(const MTLEncoder &) = delete;
        MTLEncoder &operator=(MTLEncoder &&) noexcept = delete;
        void encode_mtl_buffer(const void *buff, usize size);
        void encode_view(const ArrayDescriptor &descriptor);
        void encode_stride(const ArrayDescriptor &descriptor);

        void encode_array_buffer(const ArrayDescriptor &descriptor) {
            const ArrayBuffer &buff = descriptor.buffer();
            encode_mtl_buffer(buff.ptr(), buff.nbytes());
        }

        void set_pipeline_state(const std::string &kernel_name);
        void dispatch_threads(usize grid_nthread, usize threadgroup_nthread);
        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size);
        void wait_to_complete() { m_cmd_buff->waitUntilCompleted(); }
        double time_to_complete();
    };
} // namespace nx::runtime::metal
