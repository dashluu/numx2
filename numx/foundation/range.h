#pragma once

#include "utils.h"

namespace nx::foundation {
    struct Range {
    private:
        usize m_start;
        // Stop is exclusive
        usize m_stop;
        isize m_step;

    public:
        Range(usize start, usize stop, isize step = 1) : m_start(start), m_stop(stop), m_step(step) {}
        Range(const Range &) = default;
        Range(Range &&) noexcept = default;
        ~Range() = default;
        Range &operator=(const Range &) = default;
        Range &operator=(Range &&) noexcept = default;
        bool operator==(const Range &range) const { return m_start == range.m_start && m_stop == range.m_stop && m_step == range.m_step; }
        usize start() const { return m_start; }
        usize stop() const { return m_stop; }
        isize step() const { return m_step; }
        std::string str() const { return std::format("({},{},{})", m_start, m_stop, m_step); }
        friend std::ostream &operator<<(std::ostream &os, const Range &range) { return os << range.str(); }
    };

    using RangeVec = std::vector<Range>;
} // namespace nx::foundation
