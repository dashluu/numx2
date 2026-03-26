#include "mnist.h"
#include "numx/optim/sgd.h"

using namespace nx::core;
using namespace nx::optim;
using namespace nx::nn;

void run_basic();
void run_advanced();
void run_random();
void run_backprop();
void run_linear();
void run_multipass_with_optimizer();

int main() {
    run_backprop();
    return 0;
}

void run_basic() {
    auto x1 = full({2, 3}, 1);
    auto x2 = full({2, 3}, 2);
    auto x3 = arange({3, 3}, 1, -2);
    std::println("{}\n", x1);
    std::println("{}\n", x2);
    std::println("{}\n", x3);
    auto x4 = x1 + x2;
    std::println("{}\n", x4);
}

void run_advanced() {
    auto x1 = full({3, 3}, 1);
    auto x2 = full({3, 3}, 2);
    x1 += x2;
    std::println("{}\n", x1);
    std::println("{}\n", x2);
    auto x3 = x1.gemm(x2);
    std::println("{}\n", x3);
}

void run_random() {
    auto x1 = nx::core::normal<float>({2, 3, 4});
    auto x2 = x1.cos();
    std::println("{}\n", x1);
    std::println("{}\n", x2);
}

void run_backprop() {
    auto input = full({3, 10}, 2, &f32, DeviceKind::MPS, 0, true);
    auto output = input.sum();
    output.backward();
    std::println("{}", input.grad().value());
    std::println("{}", output.grad().value());
}

void run_linear() {
    auto input = normal<float>({3, 10});
    auto label = full({3}, 1, &nx::core::i32);
    Linear model(10, 4);
    auto logits = model(input);
    // auto loss = cross_entropy_loss(logits, label);
    auto loss = logits.sum();
    loss.backward();
    std::println("{}", model.weight().grad().value());
}

void run_multipass_with_optimizer() {
    MnistModel model;
    SGD optimizer(1);
    StateVec states;

    for (size_t i = 0; i < 3; i++) {
        auto input = normal<float>({64, 784});
        auto label = full({64}, 9, &nx::core::i32);
        auto logits = model(input);
        auto loss = cross_entropy_loss(logits, label);
        loss.backward();
        optimizer.update(model.parameters(), states);
        std::println("{}", loss);
    }
}
