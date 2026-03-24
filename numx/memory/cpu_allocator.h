#pragma once

#include "buffer_allocator.h"

namespace nx::memory {
    class CPUAllocator : public BufferAllocator {
    public:
        explicit CPUAllocator(std::size_t id) : BufferAllocator(id) {}
        std::uint8_t *alloc(std::size_t nbytes) override { return new std::uint8_t[nbytes]; }
        void free(std::uint8_t *buff) override { delete[] buff; }
    };
} // namespace nx::memory
