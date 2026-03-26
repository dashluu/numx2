#pragma once

#include "../core/array.h"

namespace nx::nn {
    using core::Array;
    using core::ArrayVec;

    class Module {
    protected:
        ArrayVec m_params;

        void add_parameter(Array &param) { m_params.push_back(param); }

    public:
        Module() = default;
        Module(const Module &) = delete;
        Module(Module &&) noexcept = delete;
        virtual ~Module() = default;
        Module &operator=(const Module &) = delete;
        Module &operator=(Module &&) noexcept = delete;
        const ArrayVec &parameters() const { return m_params; }
        ArrayVec::const_iterator begin() const { return m_params.cbegin(); }
        ArrayVec::const_iterator end() const { return m_params.cend(); }
        virtual Array forward(const Array &x) = 0;
        Array operator()(const Array &x) { return forward(x); }
    };
} // namespace nx::nn
