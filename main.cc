#include "mnist.h"
#include "numx/optim/sgd.h"

using namespace nx::core;
using namespace nx::optim;
using namespace nx::nn;

void run_basic();
void run_advanced();
void run_random();
void run_backprop_v1();
void run_backprop_v2();
void run_linear();
void run_multipass_with_vanilla_gd();
void run_multipass_with_sgd();

int main() {
    run_multipass_with_sgd();
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

void run_backprop_v1() {
    auto x1 = full({3, 10}, 2, &f32, DeviceKind::MPS, 0, true);
    auto x2 = x1.sum();
    x2.backward();
    std::println("{}\n", x1.grad().value());
    std::println("{}\n", x2.grad().value());
}

void run_backprop_v2() {
    auto x1 = full({3, 10}, 2, &f32, DeviceKind::MPS, 0, true);
    auto x2 = full({3, 10}, 3, &f32, DeviceKind::MPS, 0, true);
    auto x3 = x1 + x2;
    auto x4 = x3.sum();
    x4.backward();
    std::println("{}\n", x1.grad().value());
    std::println("{}\n", x2.grad().value());
    std::println("{}\n", x3.grad().value());
    std::println("{}\n", x4.grad().value());
}

void run_linear() {
    auto input = normal<float>({3, 10});
    auto label = full({3}, 1, &nx::core::i32);
    Linear model(10, 4);
    auto logits = model(input);
    auto loss = cross_entropy_loss(logits, label);
    loss.backward();
    std::println("{}\n", model.weight().grad().value());
    std::println("{}\n", model.bias().grad().value());
}

void run_multipass_with_vanilla_gd() {
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

void run_multipass_with_sgd() {
    MnistModel model;
    nx::optim::SGD optimizer(0.001f, 0.9f);
    StateVec states;

    for (size_t i = 0; i < model.parameters().size(); i++) {
        states.emplace_back(State());
    }

    for (size_t i = 0; i < 10; i++) {
        auto input = normal<float>({64, 784});
        auto label = full({64}, 9, &nx::core::i32);
        auto logits = model(input);
        auto loss = cross_entropy_loss(logits, label);
        loss.backward();
        optimizer.update(model.parameters(), states);
        std::println("{}", loss);
    }
}
