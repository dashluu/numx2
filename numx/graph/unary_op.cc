#include "unary_op.h"
#include "functional.h"

namespace nx::graph {
    void SqOp::backward() {
        // z = x**2
        // dx += dz * (2*x)
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(mul(m_grad, mul(graph::detach(m_operand), 2.0f)));
        }
    }

    void SqrtOp::backward() {
        // z = sqrt(x)
        // dx += dz / (2*sqrt(x))
        // dx += dz / (2*z)
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(div(m_grad, mul(detach(), 2.0f)));
        }
    }

    void NegOp::backward() {
        // z = -x
        // dx += dz * -1
        // dx -= dz
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(m_grad, false);
        }
    }

    void CopyOp::backward() {
        // z = x
        // dx += dz
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(m_grad);
        }
    }

    void ExpOp::backward() {
        // z = exp(x)
        // dx += dz * exp(x)
        // dx += dz * z
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(mul(m_grad, detach()));
        }
    }

    void LogOp::backward() {
        // z = log(x)
        // dx += dz / x
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(div(m_grad, graph::detach(m_operand)));
        }
    }

    void RecipOp::backward() {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(mul(m_grad, sq(detach())), false);
        }
    }

    void SinOp::backward() {
        // z = sin(x)
        // dx += dz * cos(x)
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(mul(m_grad, cos(graph::detach(m_operand))));
        }
    }

    void CosOp::backward() {
        // z = cos(x)
        // dx -= dz * sin(x)
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(mul(m_grad, sin(graph::detach(m_operand))), false);
        }
    }
} // namespace nx::graph
