#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nx::memory {
    class BufferAllocator {
    protected:
        std::size_t m_id;

    public:
        explicit BufferAllocator(std::size_t id) : m_id(id) {}
        BufferAllocator(const BufferAllocator &) = delete;
        BufferAllocator(BufferAllocator &&) noexcept = delete;
        virtual ~BufferAllocator() = default;
        BufferAllocator &operator=(const BufferAllocator &) = delete;
        BufferAllocator &operator=(BufferAllocator &&) noexcept = delete;
        std::size_t id() const { return m_id; }
        virtual std::uint8_t *alloc(std::size_t nbytes) = 0;
        virtual void free(std::uint8_t *buff) = 0;
    };

    using BufferAllocatorPtr = std::unique_ptr<BufferAllocator>;

    template <class T>
    concept DerivedBufferAllocator = std::is_base_of<BufferAllocator, T>::value;

    template <DerivedBufferAllocator T, class... Args>
    std::unique_ptr<T> make_buffer_allocator(Args &&...args) { return std::make_unique<T>(std::forward<Args>(args)...); }
} // namespace nx::memory
