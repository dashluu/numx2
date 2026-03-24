#pragma once

#include "array_descriptor.h"

namespace nx::foundation {
    struct ArrayIterator {
    private:
        ArrayDescriptor m_descriptor;
        std::uint8_t *m_ptr;
        isize m_counter;

    public:
        explicit ArrayIterator(const ArrayDescriptor &descriptor) : m_descriptor(descriptor) {}
        ArrayIterator(const ArrayIterator &) = delete;
        ArrayIterator(ArrayIterator &&) noexcept = delete;
        ~ArrayIterator() = default;
        ArrayIterator &operator=(const ArrayIterator &) = delete;
        ArrayIterator &operator=(ArrayIterator &&) noexcept = delete;
        bool has_next() const { return m_counter < m_descriptor.shape().get_numel(); }
        isize count() const { return m_counter; }
        void begin() { m_counter = 0; }

        std::uint8_t *next() {
            m_ptr = m_descriptor.is_contiguous() ? m_descriptor.ptr() + m_counter * m_descriptor.itemsize() : item_ptr(m_descriptor, m_counter);
            ++m_counter;
            return m_ptr;
        }
    };
} // namespace nx::foundation
