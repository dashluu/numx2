#include "graph.h"

namespace nx::graph {
    void Graph::recur_fw_sort(Op *op) {
        switch (op->kind()) {
        case PrimitiveKind::Initializer: {
            m_fw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Unary: {
            auto unary_op = static_cast<UnaryOp *>(op);
            Op *operand = unary_op->operand_raw();
            recur_fw_sort(operand);
            m_fw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Binary: {
            auto binary_op = static_cast<BinaryOp *>(op);
            Op *lhs = binary_op->lhs_raw();
            Op *rhs = binary_op->rhs_raw();
            recur_fw_sort(lhs);
            recur_fw_sort(rhs);
            m_fw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Transform: {
            auto transform_op = static_cast<TransformOp *>(op);
            Op *operand = transform_op->operand_raw();
            recur_fw_sort(operand);
            m_fw_tape.push_back(op);
            break;
        }
        default: {
            auto reduce_op = static_cast<ReduceOp *>(op);
            Op *operand = reduce_op->operand_raw();
            recur_fw_sort(operand);
            m_fw_tape.push_back(op);
            break;
        }
        }
    }

    void Graph::recur_bw_sort(Op *op) {
        switch (op->kind()) {
        case PrimitiveKind::Initializer: {
            m_bw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Unary: {
            auto unary_op = static_cast<UnaryOp *>(op);
            Op *operand = unary_op->operand_raw();
            recur_bw_sort(operand);
            m_bw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Binary: {
            auto binary_op = static_cast<BinaryOp *>(op);
            Op *lhs = binary_op->lhs_raw();
            Op *rhs = binary_op->rhs_raw();
            recur_bw_sort(lhs);
            recur_bw_sort(rhs);
            m_bw_tape.push_back(op);
            break;
        }
        case PrimitiveKind::Transform: {
            auto transform_op = static_cast<TransformOp *>(op);
            Op *operand = transform_op->operand_raw();
            recur_bw_sort(operand);
            m_bw_tape.push_back(op);
            break;
        }
        default: {
            auto reduce_op = static_cast<ReduceOp *>(op);
            Op *operand = reduce_op->operand_raw();
            recur_bw_sort(operand);
            m_bw_tape.push_back(op);
            break;
        }
        }
    }

    void Graph::fw_sort() {
        if (m_fw_tape.empty()) {
            recur_fw_sort(m_output.get());
        }
    }

    void Graph::bw_sort() {
        if (m_fw_tape.empty()) {
            throw std::runtime_error("graph has not been forwarded.");
        }

        if (m_bw_tape.empty()) {
            if (m_output->descriptor().numel() > 1) {
                throw std::runtime_error("array must be a singleton to do gradient backpropation.");
            }

            // Initialize output's gradient with 1's
            m_output->update_grad_if_enabled(ones_like(m_output));

            // Initialize gradient structure without allocating buffer memory
            // This traverses forward tape in reverse direction
            for (auto &op : std::views::reverse(m_fw_tape)) {
                if (op->is_grad_enabled()) {
                    op->backward();
                }
            }

            // Play tape backward to compute gradient
            for (auto &op : std::views::reverse(m_fw_tape)) {
                // grad is null when backward is not implemented for op or when gradient is disabled
                if (op->partial_grad() != nullptr) {
                    recur_bw_sort(op->partial_grad().get());
                }
            }
        }
    }

    void Graph::clear_grad() {
        for (auto &op : std::views::reverse(m_fw_tape)) {
            op->clear_grad();
        }
    }
} // namespace nx::graph
