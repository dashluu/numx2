#include "binary_op.h"
#include "functional.h"

namespace nx::graph {
    void AddOp::backward() {
        // In-place or not, gradient should be computed properly
        // z = x + y
        // dx += dz
        // dy += dz
        if (m_lhs->is_grad_enabled()) {
            m_lhs->update_grad(m_grad);
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->update_grad(m_grad);
        }
    }

    void SubOp::backward() {
        // z = x - y
        // dx += dz
        // dy -= dz
        if (m_lhs->is_grad_enabled()) {
            m_lhs->update_grad(m_grad);
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->update_grad(m_grad, false);
        }
    }

    void MulOp::backward() {
        // z = x * y
        // dx += dz * y
        // dy += dz * x
        // Use detach to prevent circular dependencies
        if (m_lhs->is_grad_enabled()) {
            m_lhs->update_grad(mul(m_grad, graph::detach(m_rhs)));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->update_grad(mul(m_grad, graph::detach(m_lhs)));
        }
    }

    void DivOp::backward() {
        // z = x / y
        // dx += dz * (1/y)
        // dy += dz * (-x/y**2)
        // dy -= dz * (z/y)
        // Use detach to prevent circular dependencies
        OpPtr d_rhs = graph::detach(m_rhs);

        if (m_lhs->is_grad_enabled()) {
            m_lhs->update_grad(div(m_grad, d_rhs));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->update_grad(mul(m_grad, div(detach(), d_rhs)), false);
        }
    }

    void GemmOp::backward() {
        // Transpose the last two dimensions of m_lhs and m_rhs
        // z = x @ y
        // dx += dz @ y^T
        // dy += x^T @ dz
        usize ndim = m_lhs->descriptor().ndim();

        if (m_lhs->is_grad_enabled()) {
            m_lhs->update_grad(gemm(m_grad, transpose(graph::detach(m_rhs), ndim - 2, ndim - 1)));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->update_grad(gemm(transpose(graph::detach(m_lhs), ndim - 2, ndim - 1), m_grad));
        }
    }

    void MinimumOp::backward() {
        // z = min(x, y)
        // dx += dz * (1 where x is min and 0 otherwise)
        // dy += dz * (1 where y is min and 0 otherwise)
        OpPtr d_out = detach();

        if (m_lhs->is_grad_enabled()) {
            OpPtr l_minimum = astype(eq(graph::detach(m_lhs), d_out), d_out->descriptor().dtype());
            m_lhs->update_grad(mul(m_grad, l_minimum));
        }

        if (m_rhs->is_grad_enabled()) {
            OpPtr r_minimum = astype(eq(graph::detach(m_rhs), d_out), d_out->descriptor().dtype());
            m_rhs->update_grad(mul(m_grad, r_minimum));
        }
    }

    void MaximumOp::backward() {
        // z = max(x, y)
        // dx += dz * (1 where x is max and 0 otherwise)
        // dy += dz * (1 where y is max and 0 otherwise)
        OpPtr d_out = detach();

        if (m_lhs->is_grad_enabled()) {
            OpPtr l_maximum = astype(eq(graph::detach(m_lhs), d_out), d_out->descriptor().dtype());
            m_lhs->update_grad(mul(m_grad, l_maximum));
        }

        if (m_rhs->is_grad_enabled()) {
            OpPtr r_maximum = astype(eq(graph::detach(m_rhs), d_out), d_out->descriptor().dtype());
            m_rhs->update_grad(mul(m_grad, r_maximum));
        }
    }
} // namespace nx::graph
