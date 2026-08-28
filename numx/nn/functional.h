#pragma once

#include "../core/functional.h"

namespace nx::nn {
    using core::Array;
    using foundation::i32;
    using foundation::ShapeDims;
    using foundation::usize;

    Array linear(const Array &x, const Array &weight);

    inline Array linear_with_bias(const Array &x, const Array &weight, const Array &bias) {
        return linear(x, weight) + bias;
    }

    Array onehot(const Array &x, usize num_classes);
    inline Array sigmoid(const Array &x) { return 1 / (1 + (-x).exp()); }
    inline Array tanh(const Array &x) { return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp()); }
    inline Array relu(const Array &x) { return x.maximum(0); }
    inline Array gelu_approx(const Array &x) { return 0.5 * x * (1 + tanh(std::sqrt(2 / std::numbers::pi) * (x + 0.044715 * x.pow(3)))); }
    inline Array gelu_fast(const Array &x) { return x * sigmoid(1.702 * x); }
    Array softmax(const Array &x, usize dim);
    Array cross_entropy_loss(const Array &x, const Array &y);
} // namespace nx::nn
