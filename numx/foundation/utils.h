#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <numbers>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace nx::foundation {
    using usize = std::size_t;
    using isize = std::int64_t;
    constexpr usize uone = 1;

    template <class T>
    concept NumericType = std::is_arithmetic_v<T>;

    template <class T>
    concept NumericOrBoolType = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

    template <class T>
    concept IntegerType = std::integral<T> && !std::same_as<T, bool>;

    template <class T>
    concept FloatType = std::floating_point<T>;

    static usize align_to(usize x, usize target) { return (x + target - 1) / target * target; }

    template <class T>
    std::string join(const std::vector<T> &v, const std::function<std::string(T)> &f, const std::string &sep = ",") {
        std::string s = "";

        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i > 0) s += sep;
            s += f(v[i]);
        }

        return s;
    }

    inline std::string join(const std::vector<std::string> &v, const std::string &sep = ",") {
        return join<std::string>(v, [](std::string a) { return a; }, sep);
    }

    template <NumericType T>
    std::string join_nums(const std::vector<T> &v, const std::string &sep = ",") {
        return join<T>(v, [](T a) { return std::to_string(a); }, sep);
    }
} // namespace nx::foundation
