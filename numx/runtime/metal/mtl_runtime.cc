#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "mtl_runtime.h"

namespace nx::runtime::metal {
    void MTLRuntime::run_initializer_op(Op *op) {
        switch (op->opcode()) {
        case Opcode::Full: {
            alloc_buffer(op);
            auto full_op = static_cast<graph::FullOp *>(op);
            run_full_kernel(op, full_op->constant());
            break;
        }
        case Opcode::Arange: {
            alloc_buffer(op);
            auto arange_op = static_cast<graph::ArangeOp *>(op);
            run_arange_kernel(op, arange_op->start(), arange_op->step());
            break;
        }
        case Opcode::Empty: {
            alloc_buffer(op);
            break;
        }
        default:
            break;
        }
    }

    void MTLRuntime::run_unary_op(Op *op) {
        auto unary_op = static_cast<graph::UnaryOp *>(op);
        Op *operand = unary_op->operand_raw();

        if (unary_op->in_place()) {
            share_buffer(op, operand);
        } else {
            alloc_buffer(op);
        }

        if (op->opcode() == Opcode::Copy) {
            run_copy_kernel(operand, op);
        } else {
            run_unary_kernel(operand, op);
        }
    }

    void MTLRuntime::run_binary_op(Op *op) {
        auto binary_op = static_cast<graph::BinaryOp *>(op);
        Op *lop = binary_op->lhs_raw();
        Op *rop = binary_op->rhs_raw();

        if (binary_op->type() == graph::BinaryType::Elmwise) {
            auto elmwise_op = static_cast<graph::ElmwiseBinaryOp *>(binary_op);
            if (elmwise_op->in_place()) {
                share_buffer(op, lop);
            } else {
                alloc_buffer(op);
            }
        } else {
            alloc_buffer(op);
        }

        if (binary_op->type() == graph::BinaryType::Gemm) {
            run_gemm_kernel(lop, rop, op);
        } else {
            run_binary_kernel(lop, rop, op);
        }
    }

    void MTLRuntime::run_transform_op(Op *op) {
        switch (op->opcode()) {
        case Opcode::Reshape: {
            auto reshape_op = static_cast<graph::ReshapeOp *>(op);
            Op *operand = reshape_op->operand_raw();

            if (!operand->descriptor().copy_when_reshape(reshape_op->descriptor().view())) {
                share_buffer(op, operand);
            } else {
                alloc_buffer(op);
                run_copy_kernel(operand, op);
            }

            break;
        }
        case Opcode::Slice: {
            run_simple_transform_op<graph::SliceOp>(op);
            break;
        }
        case Opcode::Broadcast: {
            run_simple_transform_op<graph::BroadcastOp>(op);
            break;
        }
        case Opcode::Permute: {
            run_simple_transform_op<graph::PermuteOp>(op);
            break;
        }
        case Opcode::Squeeze: {
            run_simple_transform_op<graph::SqueezeOp>(op);
            break;
        }
        case Opcode::Unsqueeze: {
            run_simple_transform_op<graph::UnsqueezeOp>(op);
            break;
        }
        case Opcode::Astype: {
            auto as_type_op = static_cast<graph::AstypeOp *>(op);
            Op *operand = as_type_op->operand_raw();
            alloc_buffer(op);
            run_copy_kernel(operand, op);
            break;
        }
        default:
            break;
        }
    }

    void MTLRuntime::run_reduce_op(Op *op) {
        auto reduce_op = static_cast<graph::ReduceOp *>(op);
        Op *operand = reduce_op->operand_raw();
        alloc_buffer(op);

        // Fill up array with default value
        if (reduce_op->opcode() == Opcode::Max) {
            run_full_kernel(op, reduce_op->descriptor().dtype()->min());
        } else if (reduce_op->opcode() == Opcode::Min) {
            run_full_kernel(op, reduce_op->descriptor().dtype()->max());
        } else {
            run_full_kernel(op, 0);
        }

        if (reduce_op->remaining_dims().size() == 0) {
            run_reduce_all_kernel(operand, op);
        } else {
            run_reduce_col_kernel(operand, op);
        }
    }

    void MTLRuntime::alloc_buffer(Op *op) {
        ArrayDescriptor &descriptor = op->descriptor();
        descriptor.alloc_buffer(m_ctx->memory());
    }

    void MTLRuntime::share_buffer(Op *dest, Op *src) {
        const ArrayDescriptor &src_descriptor = src->descriptor();
        ArrayDescriptor &dest_descriptor = dest->descriptor();
        dest_descriptor.share_buffer(src_descriptor);
    }
} // namespace nx::runtime::metal
