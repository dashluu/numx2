#pragma once

#include "op.h"

namespace nx::graph {
    using foundation::ShapeDims;

    struct ReduceOp : public Op {
    protected:
        OpPtr m_operand;
        ShapeDims m_remaining_dims;
        ShapeDims m_reduce_dims;

    public:
        ReduceOp(ArrayDescriptor descriptor, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : Op(std::move(descriptor)), m_operand(operand), m_remaining_dims(std::move(remaining_dims)), m_reduce_dims(std::move(reduce_dims)) {
            m_grad_enabled = operand->is_grad_enabled() || operand->is_param();
        }

        PrimitiveKind kind() const override { return PrimitiveKind::Unary; }
        std::string_view kind_str() const override { return "reduce"; }
        bool is_graph_break() const override { return false; }
        OpPtr operand() const { return m_operand; }
        Op &operand_ref() const { return *m_operand; }
        Op *operand_raw() const { return m_operand.get(); }
        const ShapeDims &remaining_dims() const { return m_remaining_dims; }
        const ShapeDims &reduce_dims() const { return m_reduce_dims; }
    };

    struct SumOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "sum";
        SumOp(ArrayDescriptor data, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : ReduceOp(std::move(data), operand, std::move(remaining_dims), std::move(reduce_dims)) {}
        Opcode opcode() const override { return Opcode::Sum; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct MaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "max";
        MaxOp(ArrayDescriptor data, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : ReduceOp(std::move(data), operand, std::move(remaining_dims), std::move(reduce_dims)) {}
        Opcode opcode() const override { return Opcode::Max; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct MinOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "min";
        MinOp(ArrayDescriptor data, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : ReduceOp(std::move(data), operand, std::move(remaining_dims), std::move(reduce_dims)) {}
        Opcode opcode() const override { return Opcode::Min; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct ArgmaxOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmax";
        ArgmaxOp(ArrayDescriptor data, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : ReduceOp(std::move(data), operand, std::move(remaining_dims), std::move(reduce_dims)) {}
        Opcode opcode() const override { return Opcode::Argmax; }
        std::string_view opname() const override { return s_opname; }
    };

    struct ArgminOp : public ReduceOp {
    public:
        inline static const std::string s_opname = "argmin";
        ArgminOp(ArrayDescriptor data, OpPtr operand, ShapeDims remaining_dims, ShapeDims reduce_dims) : ReduceOp(std::move(data), operand, std::move(remaining_dims), std::move(reduce_dims)) {}
        Opcode opcode() const override { return Opcode::Argmin; }
        std::string_view opname() const override { return s_opname; }
    };
} // namespace nx::graph
