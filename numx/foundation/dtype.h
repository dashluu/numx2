#pragma once

#include "utils.h"

namespace nx::foundation {
    enum struct DTypeKind {
        Float = 1,
        Signed = 2,
        Unsigned = 4,
        Bool = 8,
        Int = Signed | Unsigned,
        Numeric = Float | Int,
        All = Numeric | Bool
    };

    struct DType {
    protected:
        DTypeKind m_kind;
        usize m_size;

    public:
        DType(DTypeKind kind, usize size) : m_kind(kind), m_size(size) {}
        DType(const DType &) = delete;
        DType(DType &&) noexcept = delete;
        virtual ~DType() = default;
        DType &operator=(const DType &) = delete;
        DType &operator=(DType &&) noexcept = delete;
        DTypeKind kind() const { return m_kind; }
        usize size() const { return m_size; }
        bool has_kind(DTypeKind kind) const { return static_cast<int>(m_kind) & static_cast<int>(kind); }
        bool operator==(const DType &dtype) const { return m_kind == dtype.m_kind && m_size == dtype.m_size; }
        friend std::ostream &operator<<(std::ostream &os, const DType &dtype) { return os << dtype.str(); }
        std::string str() const;
        virtual std::string value(std::uint8_t *ptr) const = 0;
        virtual usize bit_cast(std::uint8_t *ptr) const = 0;
        virtual usize max() const = 0;
        virtual usize min() const = 0;
    };

    template <class T>
    struct FloatDtype : public DType {
    public:
        FloatDtype(usize size) : DType(DTypeKind::Float, size) {}

        std::string value(std::uint8_t *ptr) const override {
            T val = *reinterpret_cast<T *>(ptr);

            if (0 < val && val <= 1e-5) {
                return std::format("{:.4e}", val);
            }

            return std::format("{:.4f}", val);
        }
    };

    template <class T>
    struct IntDType : public DType {
    public:
        IntDType(DTypeKind kind, usize size) : DType(kind, size) {}
        std::string value(std::uint8_t *ptr) const override { return std::to_string(*reinterpret_cast<T *>(ptr)); }
        usize bit_cast(std::uint8_t *ptr) const override { return *reinterpret_cast<T *>(ptr); }
        usize max() const override { return std::numeric_limits<T>::max(); }
        usize min() const override { return std::numeric_limits<T>::min(); }
    };

    struct F32 : public FloatDtype<float> {
    public:
        F32() : FloatDtype<float>(4) {}
        usize bit_cast(std::uint8_t *ptr) const override { return std::bit_cast<int>(*reinterpret_cast<float *>(ptr)); }
        usize max() const override { return std::bit_cast<int>(std::numeric_limits<float>::infinity()); }
        usize min() const override { return std::bit_cast<int>(-std::numeric_limits<float>::infinity()); }
    };

    struct B8 : public DType {
    public:
        B8() : DType(DTypeKind::Bool, 1) {}
        std::string value(std::uint8_t *ptr) const override { return *ptr ? "true" : "false"; }
        usize bit_cast(std::uint8_t *ptr) const override { return *ptr; }
        usize max() const override { return std::numeric_limits<bool>::max(); }
        usize min() const override { return std::numeric_limits<bool>::min(); }
    };

    using I16 = IntDType<std::int16_t>;
    using I32 = IntDType<std::int32_t>;
    inline const I16 i16(DTypeKind::Signed, 2);
    inline const I32 i32(DTypeKind::Signed, 4);
    inline const F32 f32;
    inline const B8 b8;
    inline const std::vector<const DType *> all_dtypes = {&i32, &f32, &b8};

    inline bool is_bool(const DType *dtype) { return dtype->kind() == DTypeKind::Bool; }
    inline bool is_signed(const DType *dtype) { return dtype->kind() == DTypeKind::Signed; }
    inline bool is_unsigned(const DType *dtype) { return dtype->kind() == DTypeKind::Unsigned; }
    inline bool is_float(const DType *dtype) { return dtype->kind() == DTypeKind::Float; }
    inline bool is_int(const DType *dtype) { return dtype->has_kind(DTypeKind::Int); }
    inline bool is_numeric(const DType *dtype) { return dtype->has_kind(DTypeKind::Numeric); }

    template <NumericOrBoolType T>
    usize numeric_bitcast(const DType *dtype, T constant) {
        if (is_float(dtype)) {
            // TODO: extend this to other floating-point types
            return std::bit_cast<std::uint32_t>(static_cast<float>(constant));
        } else {
            return static_cast<usize>(constant);
        }

        return static_cast<bool>(constant);
    }

    // This only works for 32-bit data type
    std::string dtype_value(const DType *dtype, std::uint8_t *ptr);
    const DType *float_dtype_by_dtype(const DType *dtype);
} // namespace nx::foundation
