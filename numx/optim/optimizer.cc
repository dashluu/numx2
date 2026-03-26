#include "optimizer.h"

namespace nx::optim {
    void Optimizer::update(const ArrayVec &params, StateVec &states) {
        // Clear and reserve storage for parameters and gradients
        m_params.clear();
        m_grads.clear();
        m_params.reserve(params.size());
        m_grads.reserve(params.size());

        // Extract gradients from parameters and validate
        for (auto &param : params) {
            auto grad = param.grad();

            // Ensure gradient exists for each parameter
            if (!grad) {
                throw std::invalid_argument("array has no gradient for optimizer.");
            }

            // Detach parameters and gradients to prevent gradient flow
            m_params.push_back(param.detach());
            m_grads.push_back(grad.value().detach());
        }

        // Apply optimizer-specific updates to each parameter
        if (states.empty()) {
            State empty_state;

            // No optimizer state (stateless optimizers)
            for (size_t i = 0; i < m_params.size(); i++) {
                apply_single(m_params[i], m_grads[i], empty_state);
            }
        } else {
            if (m_params.size() != states.size()) {
                throw std::invalid_argument("the number of states does not match the number of parameters.");
            }

            // Use optimizer state for momentum/acceleration-based methods
            for (size_t i = 0; i < m_params.size(); i++) {
                init_single(m_params[i], states[i]);
            }

            for (size_t i = 0; i < m_params.size(); i++) {
                apply_single(m_params[i], m_grads[i], states[i]);
            }
        }

        // Evaluate all parameters to finalize updates
        for (Array &param : m_params) {
            param.eval();
        }
    }
} // namespace nx::optim
