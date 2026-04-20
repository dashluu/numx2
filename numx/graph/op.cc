#include "op.h"
#include "functional.h"

namespace nx::graph {
    void Op::zero_grad() {
        if (!m_grad) {
            m_grad = zeros_like(nonconst());
            m_partial_grad = m_grad;
        }
    }

    void Op::one_grad() {
        if (!m_grad) {
            m_grad = ones_like(nonconst());
            m_partial_grad = m_grad;
        }
    }

    void Op::clear_grad() {
        m_grad = nullptr;
        m_partial_grad = nullptr;
    }

    OpPtr Op::detach() { return graph::detach(nonconst()); }
    void Op::slice_grad(OpPtr grad, const RangeVec &ranges) { m_partial_grad = slice(grad, ranges); }

    void Op::update_grad(OpPtr grad, bool add) {
        if (add) {
            m_partial_grad = i_add(m_partial_grad, grad);
        } else {
            m_partial_grad = i_sub(m_partial_grad, grad);
        }
    }

    void Op::zero_and_update_grad(OpPtr grad, bool add) {
        zero_grad();
        update_grad(grad, add);
    }

    void Op::enable_grad(bool enabled) {
        if (!is_float(m_descriptor.dtype()) && enabled) {
            throw std::runtime_error(std::format("only floating-point arrays can have gradients but array has type {}.", m_descriptor.dtype()->str()));
        }

        m_grad_enabled = enabled;
    }
} // namespace nx::graph
