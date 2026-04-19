#include "sgd.h"

namespace nx::optim {
    void SGD::init_single(Array &param, State &state) {
        // Initialize velocity state for momentum-based updates
        Array v = zeros_like(param);
        v.eval();
        state["v"] = std::move(v);
    }

    void SGD::apply_single(Array &param, Array &grad, State &state) {
        // Apply weight decay regularization
        if (m_weight_decay != 0.0f) {
            grad += m_weight_decay * param;
        }

        // Standard gradient descent without momentum
        if (m_momentum <= 0.0f) {
            param -= m_learning_rate * grad;
            return;
        }

        // Note: state is guaranteed to be initialized before accessing
        // Accumulate velocity with momentum decay
        // Prevent the state from being a part of the computational graph
        Array &v = state["v"];
        Array velocity = v.detach();
        velocity *= m_momentum;
        // Apply dampening to reduce initial momentum effect
        velocity += m_dampening > 0.0f ? (1 - m_dampening) * grad : grad;
        // Nesterov accelerated gradient: look ahead before applying momentum
        Array grad_update = m_nesterov ? m_momentum * velocity + grad : velocity;
        // Apply parameter update
        param -= m_learning_rate * grad_update;
    }

} // namespace nx::optim
