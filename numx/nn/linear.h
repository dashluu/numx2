#pragma once

#include "functional.h"
#include "module.h"

namespace nx::nn {
    class Linear : public Module {
    private:
        Array m_weight;
        Array m_bias;
        Array m_weight_param;
        Array m_bias_param;
        bool m_has_bias;

    public:
        Linear(usize in_features, usize out_features, bool has_bias = true);
        ~Linear() = default;
        Array &weight() { return m_weight_param; }
        Array &bias() { return m_bias_param; }
        Array forward(const Array &x) override { return m_has_bias ? linear_with_bias(x, m_weight_param, m_bias_param) : linear(x, m_weight_param); }
    };
} // namespace nx::nn
