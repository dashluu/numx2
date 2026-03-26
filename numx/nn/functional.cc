#include "functional.h"

namespace nx::nn {
    Array linear(const Array &x, const Array &weight) {
        usize weight_ndim = weight.ndim();
        return x.gemm(weight.transpose(weight_ndim - 2, weight_ndim - 1));
    }

    Array onehot(const Array &x, usize num_classes) {
        if (!is_int(x.dtype())) {
            throw std::invalid_argument("array is not of type integral.");
        }

        if (num_classes <= 0) {
            num_classes = x.max().item() + 1;
        }

        Array classes = core::arange({num_classes}, 0, 1, &i32);
        return (x.unsqueeze() == classes).astype(&i32);
    }

    Array softmax(const Array &x, usize dim) {
        ShapeDims dims;
        usize ndim = x.ndim();

        if (dim < 0 || dim >= ndim) {
            dims.emplace_back(ndim - 1);
        } else {
            dims.emplace_back(dim);
        }

        Array max = x.max(dims);
        Array exp = (x - max).exp();
        Array sum_exp = exp.sum(dims);
        return exp / sum_exp;
    }

    Array cross_entropy_loss(const Array &x, const Array &y) {
        /*
        x is logits, y is target
        compute cross entropy loss -sum(y * log(softmax(x)))
        softmax(x) = exp(x) / sum(exp(x))
        log(softmax(x)) = x - log(sum(exp(x)))
        loss = -sum(y * (x - log(sum(exp(x)))))
        loss = -sum(y * x) + sum(y * log(sum(exp(x))))
        sum(y) = 1 and log(sum(exp(x))) is a scalar
        loss = -sum(y * x) + log(sum(exp(x)))
        logsumexp trick for numerical stability:
        log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        x: (*, N)
        y: (*) for labels
        max: (*, 1)
        exp: (*, N)
        sum_exp: (*, 1)
        log_sum_exp: (*, 1)
        target_onehot: (*, N)
        target_onehot * x: (*, N)
        loss: (1)
        */
        usize ndim = x.ndim();
        Array max = x.max({ndim - 1});
        Array exp = (x - max).exp();
        Array sum_exp = exp.sum({ndim - 1});
        Array log_sum_exp = sum_exp.log() + max;
        usize num_classes = x.size(ndim - 1);
        Array target_onehot = onehot(y, num_classes).astype(x.dtype());
        Array loss = -(target_onehot * x).sum({ndim - 1}) + log_sum_exp;
        return loss.mean();
    }
} // namespace nx::nn
