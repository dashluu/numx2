#include "transform_op.h"
#include "functional.h"

namespace nx::graph {
    void BroadcastOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(reshape(sum(m_grad, m_dims), m_operand_view));
        }
    }

    void ReshapeOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(reshape(m_grad, m_operand->descriptor().view()));
        }
    }

    void SliceOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->slice_grad(m_operand->grad(), m_ranges);
            m_operand->update_grad(m_grad);
        }
    }

    void PermuteOp::backward() {
        if (m_operand->is_grad_enabled()) {
            ShapeDims reverse_dims = m_grad->descriptor().shape().undo_permute_dims(m_dims);
            m_operand->update_grad(permute(m_grad, reverse_dims));
        }
    }

    void SqueezeOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(unsqueeze(m_grad, m_dims));
        }
    }

    void UnsqueezeOp::backward() {
        if (m_operand->is_grad_enabled()) {
            m_operand->update_grad(squeeze(m_grad, m_dims));
        }
    }
} // namespace nx::graph
