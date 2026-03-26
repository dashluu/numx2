#pragma once

#include "optimizer.h"

namespace nx::optim {
    class SGD : public Optimizer {
    private:
        float m_momentum;
        float m_weight_decay;
        float m_dampening;
        bool m_nesterov;

    public:
        SGD(float learning_rate, float momentum = 0.0f, float weight_decay = 0.0f, float dampening = 0.0f, bool nesterov = false) : Optimizer(learning_rate), m_momentum(momentum), m_weight_decay(weight_decay), m_dampening(dampening), m_nesterov(nesterov) {}
        void init_single(Array &param, State &state) override;
        void apply_single(Array &param, Array &grad, State &state) override;
    };
} // namespace nx::optim
