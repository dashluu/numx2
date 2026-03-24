#include "dtype.h"

namespace nx::foundation {
    std::string DType::str() const {
        std::string result;

        switch (m_kind) {
        case DTypeKind::Bool:
            result = "b";
            break;
        case DTypeKind::Signed:
            result = "i";
            break;
        case DTypeKind::Unsigned:
            result = "u";
            break;
        default:
            result = "f";
            break;
        }

        result += std::to_string(m_size * 8);
        return result;
    }

    std::string dtype_value(const DType *dtype, std::uint8_t *ptr) {
        if (is_bool(dtype)) {
            return *ptr ? "true" : "false";
        }

        if (is_signed(dtype)) {
            return std::to_string(*reinterpret_cast<std::int32_t *>(ptr));
        }

        if (is_unsigned(dtype)) {
            return std::to_string(*reinterpret_cast<std::uint32_t *>(ptr));
        }

        return std::to_string(*reinterpret_cast<float *>(ptr));
    }

    const DType *float_dtype_by_dtype(const DType *dtype) {
        if (!is_numeric(dtype)) {
            return nullptr;
        }

        return &f32;
    }
} // namespace nx::foundation
