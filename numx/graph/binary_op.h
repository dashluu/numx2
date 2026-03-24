#pragma once

#include "op.h"

namespace nx::graph {
    enum struct BinaryType {
        Elmwise,
        Cmp,
        Gemm
    };

    struct BinaryOp : public Op {
    protected:
        BinaryType m_type;
        OpPtr m_lhs;
        OpPtr m_rhs;
        bool m_in_place;

    public:
        BinaryOp(ArrayDescriptor descriptor, BinaryType type, OpPtr lhs, OpPtr rhs, bool in_place) : Op(std::move(descriptor)), m_type(type), m_lhs(lhs), m_rhs(rhs), m_in_place(in_place) {
            m_grad_enabled = lhs->is_grad_enabled() || rhs->is_grad_enabled() || lhs->is_param() || rhs->is_param();
        }

        PrimitiveKind kind() const override { return PrimitiveKind::Binary; }
        std::string_view kind_str() const override { return "binary"; }
        BinaryType type() const { return m_type; }
        bool is_graph_break() const override { return false; }
        OpPtr lhs() const { return m_lhs; }
        OpPtr rhs() const { return m_rhs; }
        Op &lhs_ref() const { return *m_lhs; }
        Op &rhs_ref() const { return *m_rhs; }
        Op *lhs_raw() const { return m_lhs.get(); }
        Op *rhs_raw() const { return m_rhs.get(); }
        bool in_place() const { return m_in_place; }
    };

    struct ElmwiseBinaryOp : public BinaryOp {
    public:
        ElmwiseBinaryOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(std::move(descriptor), BinaryType::Elmwise, lhs, rhs, in_place) {}
    };

    struct CmpOp : public BinaryOp {
    public:
        CmpOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(std::move(descriptor), BinaryType::Cmp, lhs, rhs, in_place) {}
    };

    struct AddOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "add";

        AddOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Add; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct SubOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "sub";

        SubOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Sub; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct MulOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "mul";

        MulOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Mul; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct DivOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "div";

        DivOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Div; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct EqOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "eq";

        EqOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Eq; }
        std::string_view opname() const override { return s_opname; }
    };

    struct NeqOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "neq";

        NeqOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Neq; }
        std::string_view opname() const override { return s_opname; }
    };

    struct LessOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "less";

        LessOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Less; }
        std::string_view opname() const override { return s_opname; }
    };

    struct GreaterOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "greater";

        GreaterOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Greater; }
        std::string_view opname() const override { return s_opname; }
    };

    struct LeqOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "leq";

        LeqOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Leq; }
        std::string_view opname() const override { return s_opname; }
    };

    struct GeqOp : public CmpOp {
    public:
        static constexpr std::string_view s_opname = "geq";

        GeqOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : CmpOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Geq; }
        std::string_view opname() const override { return s_opname; }
    };

    struct LogicAndOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "&&";

        LogicAndOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::LogicAnd; }
        std::string_view opname() const override { return s_opname; }
    };

    struct LogicOrOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "||";

        LogicOrOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::LogicOr; }
        std::string_view opname() const override { return s_opname; }
    };

    struct BitwiseAndOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "&";

        BitwiseAndOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::BitwiseAnd; }
        std::string_view opname() const override { return s_opname; }
    };

    struct BitwiseOrOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "|";

        BitwiseOrOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::BitwiseOr; }
        std::string_view opname() const override { return s_opname; }
    };

    struct BitwiseXorOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "^";

        BitwiseXorOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::BitwiseXor; }
        std::string_view opname() const override { return s_opname; }
    };

    struct GemmOp : public BinaryOp {
    public:
        static constexpr std::string_view s_opname = "gemm";

        GemmOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs) : BinaryOp(std::move(descriptor), BinaryType::Gemm, lhs, rhs, false) {}
        Opcode opcode() const override { return Opcode::Gemm; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct MinimumOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "minimum";

        MinimumOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Minimum; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct MaximumOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string_view s_opname = "maximum";

        MaximumOp(ArrayDescriptor descriptor, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(std::move(descriptor), lhs, rhs, in_place) {}
        Opcode opcode() const override { return Opcode::Maximum; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };
} // namespace nx::graph
