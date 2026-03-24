#pragma once

#include "../foundation/device.h"
#include "../foundation/random.h"
#include "../memory/buffer_memory.h"

namespace nx::runtime {
    using foundation::Device;
    using foundation::DevicePtr;
    using foundation::RandomKeyGenerator;
    using foundation::RandomKeyGeneratorPtr;
    using memory::BufferMemory;
    using memory::BufferMemoryPtr;

    class RuntimeContext {
    protected:
        DevicePtr m_device;
        BufferMemoryPtr m_memory;
        RandomKeyGeneratorPtr m_key_gen;

    public:
        RuntimeContext(DevicePtr device, BufferMemoryPtr memory, RandomKeyGeneratorPtr key_gen) : m_device(std::move(device)), m_memory(std::move(memory)), m_key_gen(std::move(key_gen)) {}
        RuntimeContext(const RuntimeContext &) = delete;
        RuntimeContext(RuntimeContext &&) noexcept = delete;
        virtual ~RuntimeContext() = default;
        RuntimeContext &operator=(const RuntimeContext &) = delete;
        RuntimeContext &operator=(RuntimeContext &&) noexcept = delete;
        const Device *device() const { return m_device.get(); }
        BufferMemory *memory() const { return m_memory.get(); }
        RandomKeyGenerator *key_generator() const { return m_key_gen.get(); }
    };

    using RuntimeContextPtr = std::unique_ptr<RuntimeContext>;

    template <class T>
    concept DerivedRuntimeContext = std::is_base_of<RuntimeContext, T>::value;

    template <DerivedRuntimeContext T, class... Args>
    std::unique_ptr<T> make_runtime_context(Args &&...args) { return std::make_unique<T>(std::forward<Args>(args)...); }
} // namespace nx::runtime
