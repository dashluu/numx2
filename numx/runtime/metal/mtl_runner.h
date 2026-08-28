#pragma once

#include "mtl_context.h"

namespace nx::runtime::metal {
    using foundation::ArrayBuffer;
    using foundation::ArrayDescriptor;
    using foundation::Shape;
    using foundation::ShapeStride;
    using foundation::ShapeView;
    using memory::Buffer;

    using mtl_usize = std::uint32_t;
    using mtl_isize = std::int32_t;

    class MTLRunner {
    private:
        MTLContext *m_ctx;
        MTL4::ComputeCommandEncoder *m_encoder;
        NS::SharedPtr<MTL4::ArgumentTable> m_arg_table;
        NS::SharedPtr<MTL::ResidencySet> m_residency_set;
        NS::SharedPtr<MTL::SharedEvent> m_event;
        std::vector<mtl_usize *> m_encoded_buffs;
        std::vector<MTL::Buffer *> m_mtl_buffs;
        usize m_buff_idx = 0;

    public:
        MTLRunner(RuntimeContext *ctx,
                  MTL4::ArgumentTableDescriptor *arg_table_desc,
                  MTL::ResidencySetDescriptor *residency_set_desc);
        MTLRunner(const MTLRunner &) = delete;
        MTLRunner(MTLRunner &&) noexcept = delete;
        ~MTLRunner();
        MTLRunner &operator=(const MTLRunner &) = delete;
        MTLRunner &operator=(MTLRunner &&) noexcept = delete;
        void encode_mtl_buffer(const void *buff, usize size);
        void encode_view(const Shape &shape);
        void encode_view(const ArrayDescriptor &descriptor) { encode_view(descriptor.shape()); }
        void encode_stride(const Shape &shape);
        void encode_stride(const ArrayDescriptor &descriptor) { encode_stride(descriptor.shape()); }

        void encode_array_buffer(const ArrayDescriptor &descriptor) {
            const ArrayBuffer &buff = descriptor.buffer();
            encode_mtl_buffer(buff.ptr(), buff.nbytes());
        }

        void commit(const std::string &kernel_name);
        void dispatch_threads(usize grid_nthread, usize threadgroup_nthread);
        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size);
        void dispatch_threadgroups(MTL::Size threadgroups_per_grid, MTL::Size threads_per_threadgroup);
        void run();
    };
} // namespace nx::runtime::metal
