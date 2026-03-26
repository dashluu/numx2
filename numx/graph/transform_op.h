#pragma once

#include "op.h"

namespace nx::graph {
    using foundation::DType;
    using foundation::ShapeDims;
    using foundation::ShapeView;

    struct TransformOp : public Op {
    protected:
        OpPtr m_operand;

    public:
        TransformOp(ArrayDescriptor descriptor, OpPtr operand) : Op(std::move(descriptor)), m_operand(operand) {
            m_grad_enabled = m_grad_enabled || operand->is_grad_enabled() || operand->is_param();
        }

        PrimitiveKind kind() const override { return PrimitiveKind::Transform; }
        std::string_view kind_str() const override { return "transform"; }
        bool is_graph_break() const override { return true; }
        OpPtr operand() const { return m_operand; }
        Op &operand_ref() const { return *m_operand; }
        Op *operand_raw() const { return m_operand.get(); }
    };

    struct BroadcastOp : public TransformOp {
    private:
        ShapeView m_operand_view;
        ShapeDims m_dims;

    public:
        static constexpr std::string_view s_opname = "broadcast";

        BroadcastOp(ArrayDescriptor descriptor, OpPtr operand, ShapeView operand_view, ShapeDims dims) : TransformOp(std::move(descriptor), operand), m_operand_view(std::move(operand_view)), m_dims(std::move(dims)) {}
        const ShapeView &operand_view() const { return m_operand_view; }
        const ShapeDims &dims() const { return m_dims; }
        Opcode opcode() const override { return Opcode::Broadcast; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct ReshapeOp : public TransformOp {
    public:
        static constexpr std::string_view s_opname = "reshape";
        ReshapeOp(ArrayDescriptor descriptor, OpPtr operand) : TransformOp(std::move(descriptor), operand) {}
        Opcode opcode() const override { return Opcode::Reshape; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct SliceOp : public TransformOp {
    private:
        RangeVec m_ranges;

    public:
        static constexpr std::string_view s_opname = "slice";
        SliceOp(ArrayDescriptor data, OpPtr operand, RangeVec ranges) : TransformOp(std::move(data), operand), m_ranges(std::move(ranges)) {}
        const RangeVec &ranges() const { return m_ranges; }
        Opcode opcode() const override { return Opcode::Slice; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct PermuteOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        static constexpr std::string_view s_opname = "permute";
        PermuteOp(ArrayDescriptor data, OpPtr operand, ShapeDims dims) : TransformOp(std::move(data), operand), m_dims(std::move(dims)) {}
        const ShapeDims &dims() const { return m_dims; }
        Opcode opcode() const override { return Opcode::Permute; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct SqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        static constexpr std::string_view s_opname = "squeeze";
        SqueezeOp(ArrayDescriptor data, OpPtr operand, ShapeDims dims) : TransformOp(std::move(data), operand), m_dims(std::move(dims)) {}
        const ShapeDims &dims() const { return m_dims; }
        Opcode opcode() const override { return Opcode::Squeeze; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct UnsqueezeOp : public TransformOp {
    private:
        ShapeDims m_dims;

    public:
        static constexpr std::string_view s_opname = "unsqueeze";
        UnsqueezeOp(ArrayDescriptor data, OpPtr operand, ShapeDims dims) : TransformOp(std::move(data), operand), m_dims(std::move(dims)) {}
        const ShapeDims &dims() const { return m_dims; }
        Opcode opcode() const override { return Opcode::Unsqueeze; }
        std::string_view opname() const override { return s_opname; }
        void backward() override;
    };

    struct AstypeOp : public TransformOp {
    private:
        const DType *m_dtype;

    public:
        static constexpr std::string_view s_opname = "astype";

        AstypeOp(ArrayDescriptor descriptor, OpPtr operand, const DType *dtype) : TransformOp(std::move(descriptor), operand), m_dtype(dtype) {}
        const DType *dtype() const { return m_dtype; }
        Opcode opcode() const override { return Opcode::Astype; }
        std::string_view opname() const override { return s_opname; }
    };
} // namespace nx::graph
