#include "backend.h"

#ifdef __APPLE__
#include "../runtime/metal/mtl_runtime.h"
#endif

namespace nx::core {
    Backend &Backend::instance() {
        static Backend backend;
        backend.init();
        return backend;
    }

    Runtime *Backend::runtime(usize device_id, DeviceKind device_kind) {
        std::size_t device_index = static_cast<std::size_t>(device_kind);
        return m_runtimes[device_index][device_id].get();
    }

    void Backend::init() {
        if (count_devices() > 0) {
            // This ensures backend is initialized once
            return;
        }

#ifdef __APPLE__
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        NS::Array *mtl_devices = MTL::CopyAllDevices();

        if (!mtl_devices) {
            return;
        }

        std::size_t mps_index = static_cast<std::size_t>(DeviceKind::MPS);

        while (m_runtimes.size() <= mps_index) {
            m_runtimes.emplace_back(std::vector<RuntimePtr>());
        }

        MTL::Device *mtl_device;

#ifdef PROJECT_ROOT
        const std::string project_root = PROJECT_ROOT;
#else
        const std::string project_root = ".";
#endif
        const std::string lib_path = project_root + "/build/numx/runtime/metal/kernels/kernels.metallib";

        for (NS::UInteger i = 0; i < mtl_devices->count(); ++i) {
            mtl_device = mtl_devices->object<MTL::Device>(i);
            auto device = foundation::make_device(i, DeviceKind::MPS);
            auto allocator = memory::make_buffer_allocator<CPUAllocator>(device->id());
            auto memory = memory::make_buffer_memory<SFBufferCache>(std::move(allocator));
            auto key_gen = foundation::make_random_key_generator(foundation::seed());
            auto runtime_ctx = runtime::make_runtime_context<runtime::metal::MTLContext>(mtl_device, lib_path, std::move(device), std::move(memory), std::move(key_gen));
            runtime_ctx->init_kernels();
            auto runtime = runtime::make_runtime<runtime::metal::MTLRuntime>(std::move(runtime_ctx));
            m_runtimes[mps_index].emplace_back(std::move(runtime));
            ++m_num_runtimes;
        }

        pool->release();
#endif
    }
} // namespace nx::core
