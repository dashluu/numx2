#pragma once

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

typedef uint usize;
typedef int isize;

constexpr constant uint simd_size = 32;

inline usize item_loc(uint id, usize ndim, const constant usize *shape, const constant isize *stride) {
    usize carry = id;
    usize loc = 0;

    for (isize i = ndim - 1; i >= 0; --i) {
        loc += (carry % shape[i]) * stride[i];
        carry /= shape[i];
    }

    return loc;
}

template <class T>
struct Limits {
    static T finite_min() { return metal::numeric_limits<T>::min(); }
    static T finite_max() { return metal::numeric_limits<T>::max(); }
    static T min() { return metal::numeric_limits<T>::has_infinity ? -metal::numeric_limits<T>::infinity() : finite_min(); }
    static T max() { return metal::numeric_limits<T>::has_infinity ? metal::numeric_limits<T>::infinity() : finite_max(); }
};
