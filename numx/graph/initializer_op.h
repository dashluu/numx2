#pragma once

#include "op.h"

namespace nx::graph {
    using foundation::isize;
    using foundation::usize;

    struct InitializerOp : public Op {
    public:
        explicit InitializerOp(ArrayDescriptor descriptor) : Op(std::move(descriptor)) {}
        PrimitiveKind kind() const override { return PrimitiveKind::InitializerOp; }
        std::string_view kind_str() const override { return "initializer"; }
        bool is_graph_break() const override { return true; }
    };

    struct Nop : public InitializerOp {
    public:
        static constexpr std::string_view s_opname = "nop";
        explicit Nop(ArrayDescriptor descriptor) : InitializerOp(std::move(descriptor)) {}
        Opcode opcode() const override { return Opcode::Nop; }
        std::string_view opname() const override { return s_opname; }
    };

    struct EmptyOp : public InitializerOp {
    public:
        inline static const std::string s_opname = "empty";
        explicit EmptyOp(ArrayDescriptor descriptor) : InitializerOp(std::move(descriptor)) {}
        Opcode opcode() const override { return Opcode::Empty; }
        std::string_view opname() const override { return s_opname; }
    };

    struct FullOp : public InitializerOp {
    private:
        usize m_constant;

    public:
        static constexpr std::string_view s_opname = "full";
        FullOp(ArrayDescriptor descriptor, usize constant) : InitializerOp(std::move(descriptor)), m_constant(constant) {}
        Opcode opcode() const override { return Opcode::Full; }
        std::string_view opname() const override { return s_opname; }
        usize constant() const { return m_constant; }
    };

    struct ArangeOp : public InitializerOp {
    private:
        usize m_start;
        usize m_step;

    public:
        static constexpr std::string_view s_opname = "arange";
        ArangeOp(ArrayDescriptor descriptor, usize start, usize step) : InitializerOp(std::move(descriptor)), m_start(start), m_step(step) {}
        Opcode opcode() const override { return Opcode::Arange; }
        std::string_view opname() const override { return s_opname; }
        usize start() const { return m_start; }
        usize step() const { return m_step; }
    };

    struct UniformOp : public InitializerOp {
    private:
        std::uint64_t m_key;
        usize m_low;
        usize m_high;

    public:
        static constexpr std::string_view s_opname = "uniform";
        UniformOp(ArrayDescriptor descriptor, std::uint64_t key, usize low, usize high) : InitializerOp(std::move(descriptor)), m_key(key), m_low(low), m_high(high) {}
        Opcode opcode() const override { return Opcode::Uniform; }
        std::uint64_t key() const { return m_key; }
        usize low() const { return m_low; }
        usize high() const { return m_high; }
        std::string_view opname() const override { return s_opname; }
    };
} // namespace nx::graph
