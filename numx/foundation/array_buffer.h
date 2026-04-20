#pragma once

#include "../memory/buffer_memory.h"
#include "utils.h"

namespace nx::foundation {
    using memory::Buffer;
    using memory::BufferMemory;

    struct ArrayBuffer {
    protected:
        Buffer *m_buff;
        BufferMemory *m_memory;
        bool m_is_view;

    public:
        ArrayBuffer(Buffer *buff, BufferMemory *memory, bool is_view) : m_buff(buff), m_memory(memory), m_is_view(is_view) {}

        virtual ~ArrayBuffer() {
            if (m_memory) {
                m_memory->free(m_buff);
            } else {
                delete m_buff;
            }
        }

        Buffer *buffer() const { return m_buff; }
        std::uint8_t *ptr() const { return m_buff->start(); }
        usize nbytes() const { return m_buff->nbytes(); }
        bool is_view() const { return m_is_view; }
        bool operator==(const ArrayBuffer &) = delete;
    };

    struct ArrayBufferOwner : public ArrayBuffer {
        ArrayBufferOwner(Buffer *buff, BufferMemory *memory) : ArrayBuffer(buff, memory, false) {}
        ArrayBufferOwner(const ArrayBufferOwner &) = delete;

        ArrayBufferOwner(ArrayBufferOwner &&rhs) noexcept : ArrayBuffer(rhs.m_buff, rhs.m_memory, false) {
            rhs.m_buff = nullptr;
            rhs.m_memory = nullptr;
        }

        ArrayBufferOwner &operator=(const ArrayBufferOwner &) = delete;

        ArrayBufferOwner &operator=(ArrayBufferOwner &&rhs) noexcept {
            m_buff = rhs.m_buff;
            m_memory = rhs.m_memory;
            rhs.m_buff = nullptr;
            rhs.m_memory = nullptr;
            return *this;
        }
    };

    struct ArrayBufferBorrower : public ArrayBuffer {
        explicit ArrayBufferBorrower(const ArrayBuffer &rhs) : ArrayBuffer(new Buffer(rhs.ptr(), rhs.nbytes()), nullptr, true) {}
        explicit ArrayBufferBorrower(Buffer *buff) : ArrayBuffer(buff, nullptr, true) {}
        ArrayBufferBorrower(ArrayBufferBorrower &&) noexcept = delete;
        ArrayBufferBorrower &operator=(ArrayBufferBorrower &&) noexcept = delete;

        ArrayBufferBorrower &operator=(const ArrayBuffer &rhs) {
            delete m_buff;
            m_buff = new Buffer(rhs.ptr(), rhs.nbytes());
            return *this;
        }
    };
} // namespace nx::foundation
