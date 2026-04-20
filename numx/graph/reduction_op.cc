#include "functional.h"
#include "reduce_op.h"

namespace nx::graph {
    void SumOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_and_update_grad(expand(m_grad, m_operand->descriptor().view(), m_remaining_dims, m_reduce_dims));
        }
    }

    void MaxOp::backward() {
        if (m_operand->is_grad_enabled()) {
            const ShapeView &operand_view = m_operand->descriptor().view();
            OpPtr mask = eq(graph::detach(m_operand), expand(detach(), operand_view, m_remaining_dims, m_reduce_dims));
            m_operand->zero_and_update_grad(mul(astype(mask, m_operand->descriptor().dtype()), expand(m_grad, operand_view, m_remaining_dims, m_reduce_dims)));
        }
    }

    void MinOp::backward() {
        if (m_operand->is_grad_enabled()) {
            const ShapeView &operand_view = m_operand->descriptor().view();
            OpPtr mask = eq(graph::detach(m_operand), expand(detach(), operand_view, m_remaining_dims, m_reduce_dims));
            m_operand->zero_and_update_grad(mul(astype(mask, m_operand->descriptor().dtype()), expand(m_grad, operand_view, m_remaining_dims, m_reduce_dims)));
        }
    }
} // namespace nx::graph
