#include "numx/nn/linear.h"
#include "numx/nn/module.h"

using nx::core::Array;
using nx::nn::Linear;
using nx::nn::Module;

class MnistModel : public Module {
private:
    std::unique_ptr<Linear> m_linear1;
    std::unique_ptr<Linear> m_linear2;

public:
    MnistModel() {
        m_linear1 = std::make_unique<Linear>(784, 128);
        m_linear2 = std::make_unique<Linear>(128, 10);
        const auto &params1 = m_linear1->parameters();
        const auto &params2 = m_linear2->parameters();
        m_params.reserve(params1.size() + params2.size());
        m_params.insert(m_params.end(), params1.begin(), params1.end());
        m_params.insert(m_params.end(), params2.begin(), params2.end());
    }

    MnistModel(const MnistModel &) = delete;
    MnistModel(MnistModel &&) noexcept = delete;
    MnistModel &operator=(const MnistModel &) = delete;
    MnistModel &operator=(MnistModel &&) noexcept = delete;

    Array forward(const Array &x) override {
        Array x1 = (*m_linear1)(x);
        Array x2 = nx::nn::relu(x1);
        Array x3 = (*m_linear2)(x2);
        return x3;
    }
};
