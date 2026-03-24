#pragma once

#include "buffer.h"
#include <memory>

namespace nx::memory {
    class BufferMemory {
    protected:
        BufferAllocatorPtr m_allocator;

    public:
        explicit BufferMemory(BufferAllocatorPtr allocator) : m_allocator(std::move(allocator)) {}
        BufferMemory(const BufferMemory &) = delete;
        BufferMemory(BufferMemory &&) noexcept = delete;
        virtual ~BufferMemory() = default;
        BufferMemory &operator=(const BufferMemory &) = delete;
        BufferMemory &operator=(BufferMemory &&) noexcept = delete;
        virtual Buffer *alloc(std::size_t nbytes) = 0;
        virtual void free(Buffer *buff) = 0;
    };

    using BufferMemoryPtr = std::unique_ptr<BufferMemory>;

    template <class T>
    concept DerivedBufferMemory = std::is_base_of<BufferMemory, T>::value;

    template <DerivedBufferMemory T, class... Args>
    std::unique_ptr<T> make_buffer_memory(Args &&...args) { return std::make_unique<T>(std::forward<Args>(args)...); }
} // namespace nx::memory
