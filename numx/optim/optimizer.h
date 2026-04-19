#pragma once

#include "../core/functional.h"

namespace nx::optim {
    using core::Array;
    using core::ArrayVec;
    using core::State;
    using core::StateVec;

    class Optimizer {
    protected:
        float m_learning_rate;
        ArrayVec m_params;
        ArrayVec m_grads;

    public:
        explicit Optimizer(float learning_rate) : m_learning_rate(learning_rate) {}
        Optimizer(const Optimizer &) = delete;
        Optimizer(Optimizer &&) noexcept = delete;
        virtual ~Optimizer() = default;
        Optimizer &operator=(const Optimizer &) = delete;
        Optimizer &operator=(Optimizer &&) noexcept = delete;
        virtual void init_single(Array &param, State &state) = 0;
        virtual void apply_single(Array &param, Array &grad, State &state) = 0;
        void update(const ArrayVec &params, StateVec &states);
    };
} // namespace nx::optim
