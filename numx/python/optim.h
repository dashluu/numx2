#pragma once

#include "bind.h"

namespace nx::bind {
    struct PyOptimizer : nxo::Optimizer {
        NB_TRAMPOLINE(nxo::Optimizer, 1);

        void init_single(nxc::Array &param, nxo::State &state) override {
            NB_OVERRIDE_PURE(init_single, param, state);
        }

        void apply_single(nxc::Array &param, nxc::Array &grad, nxo::State &state) override {
            NB_OVERRIDE_PURE(apply_single, param, grad, state);
        }
    };
} // namespace nx::bind
