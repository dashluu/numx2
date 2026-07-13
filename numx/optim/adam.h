#pragma once

#include "optimizer.h"

namespace nx::optim {
    class Adam : public Optimizer {
    private:
        float m_beta1;
        float m_beta2;
        float m_epsilon;
        bool m_bias_correction;

    public:
        Adam(float learning_rate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, bool bias_correction = false) : Optimizer(learning_rate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_bias_correction(bias_correction) {}
        void init_single(Array &param, State &state) override;
        void apply_single(Array &param, Array &grad, State &state) override;
    };
} // namespace nx::optim
