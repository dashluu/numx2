#include "reduce.h"

template <class T, class R>
kernel void simd_gevv(
    const constant usize *offset [[buffer(0)]],
    const constant usize *l_shape [[buffer(1)]],
    const constant usize *r_shape [[buffer(2)]],
    const device T *lhs [[buffer(3)]],
    const device T *rhs [[buffer(4)]],
    device metal::_atomic<R> *output [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    usize K = l_shape[0];
    T product = gid < K ? lhs[offset[0] + gid] * rhs[offset[1] + gid] : T(0);
    T sum = metal::simd_sum(product);
    threadgroup T ldata[simd_size];

    if (simd_per_group > 1) {
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = sum;
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        sum =  (lid < simd_per_group) ? ldata[lid] : T(0);
        sum = metal::simd_sum(sum);
    }

    if (lid == 0) {
        AtomicSum()(output + offset[2], sum);
    }
}

template <class T, class R>
kernel void strided_simd_gevv(
    const constant usize *offset [[buffer(0)]],
    const constant usize *l_shape [[buffer(1)]],
    const constant usize *r_shape [[buffer(2)]],
    const constant isize *l_stride [[buffer(3)]],
    const constant isize *r_stride [[buffer(4)]],
    const device T *lhs [[buffer(5)]],
    const device T *rhs [[buffer(6)]],
    device metal::_atomic<R> *output [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    usize K = l_shape[0];
    usize l_loc = item_loc(gid, 1, l_shape, l_stride);
    usize r_loc = item_loc(gid, 1, r_shape, r_stride);
    T product = gid < K ? lhs[offset[0] + l_loc] * rhs[offset[1] + r_loc] : T(0);
    T sum = metal::simd_sum(product);
    threadgroup T ldata[simd_size];

    if (simd_per_group > 1) {
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = sum;
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        sum =  (lid < simd_per_group) ? ldata[lid] : T(0);
        sum = metal::simd_sum(sum);
    }

    if (lid == 0) {
        AtomicSum()(output + offset[2], sum);
    }
}

#define def_simd_gevv(dtype, T, R)    \
template [[host_name("simd_gevv_" #dtype)]] [[kernel]] decltype(simd_gevv<T, R>) simd_gevv<T, R>;                          \
template [[host_name("strided_simd_gevv_" #dtype)]] [[kernel]] decltype(strided_simd_gevv<T, R>) strided_simd_gevv<T, R>;

def_simd_gevv(f32, float, float);
def_simd_gevv(i32, int, int);
