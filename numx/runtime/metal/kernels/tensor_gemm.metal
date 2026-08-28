#include "utils.h"

template <class T, class R, uint TileSize, uint Simdgroups>
void tile_gemm2d(threadgroup T *lhs, threadgroup T *rhs, threadgroup R *output)
{
    metal::array<int, 2> stride = {1, TileSize};
    metal::tensor<threadgroup T, metal::dextents<int, 2>, metal::tensor_inline> l_tensor(lhs, metal::dextents<int, 2>(TileSize, TileSize), stride);
    metal::tensor<threadgroup T, metal::dextents<int, 2>, metal::tensor_inline> r_tensor(rhs, metal::dextents<int, 2>(TileSize, TileSize), stride);
    metal::tensor<threadgroup R, metal::dextents<int, 2>, metal::tensor_inline> out_tensor(output, metal::dextents<int, 2>(TileSize, TileSize), stride);
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(TileSize, TileSize, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    // Create a 2D matrix multiplication operation with some SIMD groups.
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<Simdgroups>> matmul_op;
    // Capture the matrix multiplication output in a cooperative tensor.
    auto cooperative_tensor = matmul_op.template get_destination_cooperative_tensor<decltype(l_tensor), decltype(r_tensor), T>();
    cooperative_tensor.load(out_tensor);
    // Run the matrix multiplication.
    matmul_op.run(l_tensor, r_tensor, cooperative_tensor);
    // Store the results to the destination slice.
    cooperative_tensor.store(out_tensor);
}

template <class T, uint TileSize, uint SimdSize>
void load_tile_tensor2d(
    const usize offset,
    const usize rows,
    const usize cols,
    const constant isize *stride,
    threadgroup T *threadgroup_buff,
    device T *device_buff,
    uint2 lid,
    uint2 tid
)
{
    for (uint i = 0; i < SimdSize; ++i) {
        for (uint j = 0; j < SimdSize; ++j) {
            uint ly = lid.y * SimdSize + i;
            uint lx = lid.x * SimdSize + j;
            uint gy = tid.y * TileSize + ly;
            uint gx = tid.x * TileSize + lx;
            uint threadgroup_loc = ly * TileSize + lx;

            if (gy < rows && gx < cols) {
                isize device_loc = gy * stride[0] + gx * stride[1];
                threadgroup_buff[threadgroup_loc] = device_buff[offset + device_loc];
            } else {
                threadgroup_buff[threadgroup_loc] = 0;
            }
        }
    }
}

template <class T, uint TileSize, uint SimdSize>
void load_tile_tensor3d(
    const usize ndim,
    const usize offset,
    const usize batch_loc,
    const usize rows,
    const usize cols,
    const constant isize *stride,
    threadgroup T *threadgroup_buff,
    device T *device_buff,
    uint2 lid,
    uint2 tid
)
{
    for (uint i = 0; i < SimdSize; ++i) {
        for (uint j = 0; j < SimdSize; ++j) {
            uint ly = lid.y * SimdSize + i;
            uint lx = lid.x * SimdSize + j;
            uint gy = tid.y * TileSize + ly;
            uint gx = tid.x * TileSize + lx;
            uint threadgroup_loc = ly * TileSize + lx;

            if (gy < rows && gx < cols) {
                isize device_loc = batch_loc + gy * stride[ndim - 2] + gx * stride[ndim - 1];
                threadgroup_buff[threadgroup_loc] = device_buff[offset + device_loc];
            } else {
                threadgroup_buff[threadgroup_loc] = 0;
            }
        }
    }
}

template <class T, uint TileSize, uint SimdSize>
void store_tile_tensor2d(
    const usize offset,
    const usize rows,
    const usize cols,
    const thread isize *stride,
    threadgroup T *threadgroup_buff,
    device T *device_buff,
    uint2 lid,
    uint2 tid
)
{
    for (uint i = 0; i < SimdSize; ++i) {
        for (uint j = 0; j < SimdSize; ++j) {
            uint ly = lid.y * SimdSize + i;
            uint lx = lid.x * SimdSize + j;
            uint gy = tid.y * TileSize + ly;
            uint gx = tid.x * TileSize + lx;
            uint threadgroup_loc = ly * TileSize + lx;

            if (gy < rows && gx < cols) {
                isize device_loc = gy * stride[0] + gx * stride[1];
                device_buff[offset + device_loc] = threadgroup_buff[threadgroup_loc];
            }
        }
    }
}

template <class T, uint TileSize, uint SimdSize>
void store_tile_tensor3d(
    const usize offset,
    const usize batch,
    const usize rows,
    const usize cols,
    const thread isize *stride,
    threadgroup T *threadgroup_buff,
    device T *device_buff,
    uint2 lid,
    uint2 tid
)
{
    for (uint i = 0; i < SimdSize; ++i) {
        for (uint j = 0; j < SimdSize; ++j) {
            uint ly = lid.y * SimdSize + i;
            uint lx = lid.x * SimdSize + j;
            uint gy = tid.y * TileSize + ly;
            uint gx = tid.x * TileSize + lx;
            uint threadgroup_loc = ly * TileSize + lx;

            if (gy < rows && gx < cols) {
                isize device_loc = batch * stride[0] + gy * stride[1] + gx * stride[2];
                device_buff[offset + device_loc] = threadgroup_buff[threadgroup_loc];
            }
        }
    }
}

template <class T, class R, uint TileSize, uint Simdgroups, uint SimdSize>
kernel void tensor_gemm2d(
    const constant usize *offset [[buffer(0)]],
    const constant usize *l_shape [[buffer(1)]],
    const constant usize *r_shape [[buffer(2)]],
    const constant isize *l_stride [[buffer(3)]],
    const constant isize *r_stride [[buffer(4)]],
    device T *lhs [[buffer(5)]],
    device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]]
)
{
    usize M = l_shape[0], K = l_shape[1], N = r_shape[1];
    usize inner_tile_cols = (K + TileSize - 1) / TileSize;
    usize result_tile_cols = (N + TileSize - 1) / TileSize;
    uint simd_cols = TileSize / SimdSize;
    uint lidy = lid / simd_cols;
    uint lidx = lid % simd_cols;
    uint tidy = tid / result_tile_cols;
    uint tidx = tid % result_tile_cols;
    uint2 lid2(lidx, lidy);
    isize out_stride[] = {static_cast<isize>(N), 1};
    threadgroup T l_buff[TileSize * TileSize];
    threadgroup T r_buff[TileSize * TileSize];
    threadgroup R out_buff[TileSize * TileSize];

    for (usize i = 0; i < inner_tile_cols; ++i) {
        load_tile_tensor2d<T, TileSize, SimdSize>(offset[0], M, K, l_stride, l_buff, lhs, lid2, uint2(i, tidy));
        load_tile_tensor2d<T, TileSize, SimdSize>(offset[1], K, N, r_stride, r_buff, rhs, lid2, uint2(tidx, i));
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        tile_gemm2d<T, R, TileSize, Simdgroups>(l_buff, r_buff, out_buff);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    store_tile_tensor2d<R, TileSize, SimdSize>(offset[2], M, N, out_stride, out_buff, output, lid2, uint2(tidx, tidy));
}

template <class T, class R, uint TileSize, uint Simdgroups, uint SimdSize>
kernel void tensor_gemm3d(
    const constant usize &ndim [[buffer(0)]],
    const constant usize *offset [[buffer(1)]],
    const constant usize *l_shape [[buffer(2)]],
    const constant usize *r_shape [[buffer(3)]],
    const constant isize *l_stride [[buffer(4)]],
    const constant isize *r_stride [[buffer(5)]],
    const constant usize &threagroups_per_batch [[buffer(6)]],
    device T *lhs [[buffer(7)]],
    device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]]
)
{
    usize M = l_shape[ndim - 2], K = l_shape[ndim - 1], N = r_shape[ndim - 1];
    usize inner_tile_cols = (K + TileSize - 1) / TileSize;
    usize result_tile_cols = (N + TileSize - 1) / TileSize;
    uint simd_cols = TileSize / SimdSize;
    uint lidy = lid / simd_cols;
    uint lidx = lid % simd_cols;
    uint bidy = tid / threagroups_per_batch;
    uint bidx = tid % threagroups_per_batch;
    uint tidy = bidx / result_tile_cols;
    uint tidx = bidx % result_tile_cols;
    uint2 lid2(lidx, lidy);
    isize out_stride[] = {static_cast<isize>(M * N), static_cast<isize>(N), 1};
    usize l_batch_loc = item_loc(bidy, ndim - 2, l_shape, l_stride);
    usize r_batch_loc = item_loc(bidy, ndim - 2, r_shape, r_stride);
    threadgroup T l_buff[TileSize * TileSize];
    threadgroup T r_buff[TileSize * TileSize];
    threadgroup R out_buff[TileSize * TileSize];

    for (usize i = 0; i < inner_tile_cols; ++i) {
        load_tile_tensor3d<T, TileSize, SimdSize>(ndim, offset[0], l_batch_loc, M, K, l_stride, l_buff, lhs, lid2, uint2(i, tidy));
        load_tile_tensor3d<T, TileSize, SimdSize>(ndim, offset[1], r_batch_loc, K, N, r_stride, r_buff, rhs, lid2, uint2(tidx, i));
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        tile_gemm2d<T, R, TileSize, Simdgroups>(l_buff, r_buff, out_buff);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    store_tile_tensor3d<R, TileSize, SimdSize>(offset[2], bidy, M, N, out_stride, out_buff, output, lid2, uint2(tidx, tidy));
}

#define def_tensor_gemm(dtype, T, R)    \
template [[host_name("tensor_gemm2d_" #dtype)]] [[kernel]] decltype(tensor_gemm2d<T, R, 32, 2, 4>) tensor_gemm2d<T, R, 32, 2, 4>;   \
template [[host_name("tensor_gemm3d_" #dtype)]] [[kernel]] decltype(tensor_gemm3d<T, R, 32, 2, 4>) tensor_gemm3d<T, R, 32, 2, 4>;

def_tensor_gemm(f32, float, float)
