#pragma once

#include "buffer.h"
#include <chrono>

namespace nx::memory {
    struct BufferSnapshot {
    private:
        using clock = std::chrono::steady_clock;
        using time_point = clock::time_point;
        const std::uint8_t *m_ptr;
        std::size_t m_nbytes;
        time_point m_alloc_time;
        std::optional<time_point> m_free_time;

    public:
        explicit BufferSnapshot(Buffer *buff) : m_ptr(buff->start()), m_nbytes(buff->nbytes()) {
            m_alloc_time = clock::now();
        }

        BufferSnapshot(const BufferSnapshot &) = default;
        BufferSnapshot(BufferSnapshot &&) noexcept = default;
        ~BufferSnapshot() = default;
        BufferSnapshot &operator=(const BufferSnapshot &) = default;
        BufferSnapshot &operator=(BufferSnapshot &&) noexcept = default;
        const std::uint8_t *ptr() const { return m_ptr; }
        std::size_t nbytes() const { return m_nbytes; }
        const time_point &alloc_time() const { return m_alloc_time; }
        const std::optional<time_point> &free_time() const { return m_free_time; }
        bool is_alive() const { return !m_free_time.has_value(); }

        void stop() {
            if (!m_free_time.has_value()) {
                m_free_time = clock::now();
            }
        }

        std::chrono::microseconds elapsed() const {
            time_point free_time = m_free_time.value_or(clock::now());
            return std::chrono::duration_cast<std::chrono::microseconds>(free_time - m_alloc_time);
        }
    };
} // namespace nx::memory
