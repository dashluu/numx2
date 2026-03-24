#pragma once

#include "buffer_allocator.h"
#include <cassert>

namespace nx::memory {
    struct Buffer {
    protected:
        BufferAllocator *m_allocator;
        std::uint8_t *m_ptr;
        std::size_t m_nbytes;

    public:
        Buffer(BufferAllocator *allocator, std::size_t nbytes) : m_allocator(allocator), m_nbytes(nbytes) { m_ptr = allocator->alloc(nbytes); }

        Buffer(std::uint8_t *ptr, std::size_t nbytes) : m_allocator(nullptr), m_ptr(ptr), m_nbytes(nbytes) {}
        Buffer(const Buffer &) = delete;
        Buffer(Buffer &&) noexcept = delete;

        virtual ~Buffer() {
            if (m_allocator) {
                m_allocator->free(m_ptr);
            }

            m_ptr = nullptr;
        }

        Buffer &operator=(const Buffer &) = delete;
        Buffer &operator=(Buffer &&) noexcept = delete;
        std::uint8_t *start() const { return m_ptr; }
        std::uint8_t *end() const { return m_ptr + m_nbytes; }
        std::size_t nbytes() const { return m_nbytes; }
        BufferAllocator *allocator() const { return m_allocator; }
        bool is_allocated() const { return m_allocator != nullptr; }
    };
} // namespace nx::memory
