#pragma once

#include "primitive.h"

namespace nx::graph {
    using foundation::RangeVec;
    using foundation::usize;

    enum struct Opcode {
        Nop,
        Empty,
        Arange,
        Uniform,
        Normal,
        Full,
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Eq,
        Neq,
        Greater,
        Less,
        Leq,
        Geq,
        LogicAnd,
        LogicOr,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        Minimum,
        Maximum,
        Gemm,
        Sq,
        Sqrt,
        Neg,
        LogicNot,
        BitwiseNot,
        Copy,
        Exp,
        Log,
        Recip,
        Sin,
        Cos,
        Reshape,
        Permute,
        Broadcast,
        Squeeze,
        Unsqueeze,
        Slice,
        Sum,
        Max,
        Min,
        Argmax,
        Argmin,
        Astype
    };

    struct Op;
    using OpPtr = std::shared_ptr<Op>;
    class Graph;

    struct Op : public Primitive {
    private:
        friend class Graph;

    protected:
        ArrayDescriptor m_descriptor;
        bool m_grad_enabled;
        OpPtr m_grad = nullptr;
        OpPtr m_partial_grad = nullptr;

        OpPtr nonconst() const { return nonconst_primitive<Op>(shared_from_this()); }

    public:
        explicit Op(ArrayDescriptor descriptor) : Primitive(), m_descriptor(std::move(descriptor)) {
            m_grad_enabled = descriptor.is_param();
        }

        ArrayDescriptor &descriptor() { return m_descriptor; }
        virtual Opcode opcode() const = 0;
        virtual std::string_view opname() const = 0;
        bool is_grad_enabled() const { return m_grad_enabled; }
        bool is_param() const { return m_descriptor.is_param(); }
        OpPtr grad() const { return m_grad; }
        OpPtr partial_grad() const { return m_partial_grad; }
        virtual bool is_graph_break() const = 0;
        OpPtr detach();
        void clear_grad();
        void zero_grad();
        void one_grad();
        void slice_grad(OpPtr grad, const RangeVec &ranges);
        void update_grad(OpPtr grad, bool add = true);
        void zero_and_update_grad(OpPtr grad, bool add = true);
        void enable_grad(bool enabled);
    };
} // namespace nx::graph
