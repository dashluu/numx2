#pragma once

#include "utils.h"

namespace nx::foundation {
    class IncompatShapesForOp : public std::invalid_argument {
    public:
        IncompatShapesForOp(std::string_view opname, std::string_view l_view_str, std::string_view r_view_str) : std::invalid_argument(std::format("cannot run operator {} on incompatible shapes {} and {}.", opname, l_view_str, r_view_str)) {}
    };

    class IncompatDTypesForOp : public std::invalid_argument {
    public:
        IncompatDTypesForOp(std::string_view opname, std::string_view l_dtype_str, std::string_view r_dtype_str) : std::invalid_argument(std::format("cannot run operator {} on incompatible data types {} and {}.", opname, l_dtype_str, r_dtype_str)) {}
    };

    class IncompatDTypeForOp : public std::invalid_argument {
    public:
        IncompatDTypeForOp(std::string_view opname, std::string_view dtype_str) : std::invalid_argument(std::format("cannot run operator {} on incompatible data type {}.", opname, dtype_str)) {}
    };

    class IncompatDevicesForOp : public std::invalid_argument {
    public:
        IncompatDevicesForOp(std::string_view opname, std::string_view l_device_str, std::string_view r_device_str) : std::invalid_argument(std::format("cannot run operator {} on incompatible devices {} and {}.", opname, l_device_str, r_device_str)) {}
    };

    class IncompatDTypeForRandomFunction : public std::invalid_argument {
    public:
        IncompatDTypeForRandomFunction(std::string_view function_name, std::string_view expected_dtype_str, std::string_view actual_dtype_str) : std::invalid_argument(std::format("{}() only accepts {} data type but got {}.", function_name, expected_dtype_str, actual_dtype_str)) {}
    };

    class NanobindInvalidArgumentType : public std::invalid_argument {
    public:
        NanobindInvalidArgumentType(std::string_view expected_type_name, std::string_view actual_type_name) : std::invalid_argument(std::format("expected an argument of type {} but received an argument of type {}.", expected_type_name, actual_type_name)) {}
    };
} // namespace nx::foundation
