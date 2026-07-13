#include "adam.h"

namespace nx::optim {
    void Adam::init_single(Array &param, State &state) {
        Array m = zeros_like(param);
        Array v = zeros_like(param);
        m.eval();
        v.eval();
        state["m"] = std::move(m);
        state["v"] = std::move(v);
    }

    void Adam::apply_single(Array &param, Array &grad, State &state) {
        Array m = state["m"].detach();
        Array v = state["v"].detach();
        m = m_beta1 * m + (1 - m_beta1) * grad;
        v = m_beta2 * v + (1 - m_beta2) * grad.sq();

        if (m_bias_correction) {
            Array m_normalized = m / (1 - std::pow(m_beta1, m_step));
            Array v_normalized = v / (1 - std::pow(m_beta2, m_step));
            param -= m_learning_rate * m_normalized / (v_normalized.sqrt() + m_epsilon);
        } else {
            param -= m_learning_rate * m / (v.sqrt() + m_epsilon);
        }
    }

} // namespace nx::optim
