#pragma once

#include "../foundation/array_descriptor.h"

namespace nx::graph {
    using foundation::ArrayDescriptor;

    enum struct PrimitiveKind {
        Initializer,
        Unary,
        Binary,
        Transform,
        Reduce,
        FusedKernel
    };

    struct Primitive : public std::enable_shared_from_this<Primitive> {
    public:
        explicit Primitive() {}
        Primitive(const Primitive &) = delete;
        Primitive(Primitive &&) = delete;
        virtual ~Primitive() = default;
        virtual PrimitiveKind kind() const = 0;
        virtual std::string_view kind_str() const = 0;
        virtual void backward() {}
    };

    using PrimitivePtr = std::shared_ptr<Primitive>;

    template <class T>
    concept DerivedPrimitive = std::is_base_of<Primitive, T>::value;

    template <DerivedPrimitive T, class... Args>
    std::shared_ptr<T> make_primitive(Args &&...args) { return std::make_shared<T>(std::forward<Args>(args)...); }

    template <DerivedPrimitive T>
    std::shared_ptr<T> cast_primitive(PrimitivePtr primitive) { return std::static_pointer_cast<T>(primitive); }

    template <DerivedPrimitive T>
    std::shared_ptr<T> nonconst_primitive(std::shared_ptr<const Primitive> primitive) { return std::const_pointer_cast<T>(std::static_pointer_cast<const T>(primitive)); }
} // namespace nx::graph
