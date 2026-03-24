#pragma once

#include "op.h"

namespace nx::graph {
    struct UnaryOp : public Op {
    protected:
        OpPtr m_operand;
        bool m_in_place;

    public:
        UnaryOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : Op(std::move(descriptor)), m_operand(operand), m_in_place(in_place) {
            m_grad_enabled = operand->is_grad_enabled() || operand->is_param();
        }

        PrimitiveKind kind() const override { return PrimitiveKind::Unary; }
        std::string_view kind_str() const override { return "unary"; }
        bool is_graph_break() const override { return false; }
        OpPtr operand() const { return m_operand; }
        Op &operand_ref() const { return *m_operand; }
        Op *operand_raw() const { return m_operand.get(); }
        bool in_place() const { return m_in_place; }
    };

    struct SqOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "sq";

        SqOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Sq; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct SqrtOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "sqrt";
        SqrtOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Sqrt; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct NegOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "neg";
        NegOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Neg; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct LogicNot : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "!";
        LogicNot(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::LogicNot; }
        std::string_view opname() const override { return s_opname; }
    };

    struct BitwiseNot : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "~";
        BitwiseNot(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::BitwiseNot; }
        std::string_view opname() const override { return s_opname; }
    };

    struct CopyOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "copy";
        CopyOp(ArrayDescriptor descriptor, OpPtr operand) : UnaryOp(std::move(descriptor), operand, false) {}
        Opcode opcode() const override { return Opcode::Copy; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct ExpOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "exp";
        ExpOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Exp; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct LogOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "log";

        LogOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Log; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct RecipOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "recip";
        RecipOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Recip; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct SinOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "sin";

        SinOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Sin; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct CosOp : public UnaryOp {
    public:
        static constexpr std::string_view s_opname = "cos";

        CosOp(ArrayDescriptor descriptor, OpPtr operand, bool in_place) : UnaryOp(std::move(descriptor), operand, in_place) {}
        Opcode opcode() const override { return Opcode::Cos; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };
} // namespace nx::graph
