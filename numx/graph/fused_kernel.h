#pragma once

#include "op.h"

namespace nx::graph {
    struct FusedKernel : public Primitive {
    private:
        std::vector<Op *> m_ops;

    public:
        void add(Op *op) { m_ops.push_back(op); }
        std::vector<Op *>::const_iterator begin() const { return m_ops.cbegin(); }
        std::vector<Op *>::const_iterator end() const { return m_ops.cend(); }
    };
} // namespace nx::graph
