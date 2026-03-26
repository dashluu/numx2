#include "linear.h"

namespace nx::nn {
    Linear::Linear(usize in_features, usize out_features, bool has_bias) : m_has_bias(has_bias) {
        m_weight = core::kaiming_uniform({out_features, in_features});
        m_weight.eval();
        m_weight_param = m_weight.detach(true);
        add_parameter(m_weight_param);

        if (has_bias) {
            auto [fan_in, fan_out] = compute_fan_in_and_fan_out(m_weight_param);
            float bound = 1.0f / std::sqrt(fan_in);
            m_bias = core::uniform({out_features}, -bound, bound);
            m_bias.eval();
            m_bias_param = m_bias.detach(true);
            add_parameter(m_bias_param);
        }
    }
} // namespace nx::nn
