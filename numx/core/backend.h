#pragma once

#include "../foundation/device.h"
#include "../foundation/random.h"
#include "../memory/cpu_allocator.h"
#include "../memory/sf_buffer_cache.h"
#include "../runtime/runtime.h"
#include <print>

namespace nx::core {
    using foundation::DeviceKind;
    using foundation::DevicePtr;
    using foundation::RandomKeyGenerator;
    using foundation::RandomKeyGeneratorPtr;
    using foundation::usize;
    using memory::BufferAllocatorPtr;
    using memory::BufferMemoryPtr;
    using memory::CPUAllocator;
    using memory::SFBufferCache;
    using runtime::Runtime;
    using runtime::RuntimePtr;

    class Backend {
    private:
        std::vector<std::vector<RuntimePtr>> m_runtimes;
        std::size_t m_num_runtimes = 0;

        Backend() = default;
        void init();
        std::size_t count_devices() const { return m_num_runtimes; }

    public:
        static Backend &instance();
        Runtime *runtime(usize device_id, DeviceKind device_kind);
    };
} // namespace nx::core
