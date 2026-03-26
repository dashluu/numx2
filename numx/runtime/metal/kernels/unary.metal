#include "unary.h"

template <class Op, class T, class R>
kernel void unary(
    const constant usize *offset [[buffer(0)]],
    const device T *input [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[1] + id] = Op()(input[offset[0] + id]);
}

template <class Op, class T, class R>
kernel void strided_unary(
    const constant usize &ndim [[buffer(0)]],
    const constant usize *offset [[buffer(1)]],
    const constant usize *shape [[buffer(2)]],
    const constant isize *in_stride [[buffer(3)]],
    const constant isize *out_stride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    const device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    usize in_loc = strided[0] ? item_loc(id, ndim, shape, in_stride) : id;
    usize out_loc = strided[1] ? item_loc(id, ndim, shape, out_stride) : id;
    output[offset[1] + out_loc] = Op()(input[offset[0] + in_loc]);
}

#define def_unary_kernels(opname, op, dtype, T, R)      \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, R>) unary<op, T, R>;                                    \
template [[host_name("strided_" #opname "_" #dtype)]] [[kernel]] decltype(strided_unary<op, T, R>) strided_unary<op, T, R>;

#define def_unary_float_kernels(opname, op, dtype, T)   \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, float>) unary<op, T, float>;                            \
template [[host_name("strided_" #opname "_" #dtype)]] [[kernel]] decltype(strided_unary<op, T, float>) strided_unary<op, T, float>;

#define def_logic_kernels(opname, op)                   \
template [[host_name(#opname "_b8")]] [[kernel]] decltype(unary<op, bool, bool>) unary<op, bool, bool>;                             \
template [[host_name("strided_" #opname "_b8")]] [[kernel]] decltype(strided_unary<op, bool, bool>) strided_unary<op, bool, bool>;

#define def_unary_float(opname, op)                     \
def_unary_float_kernels(opname, op, f32, float);        \
def_unary_float_kernels(opname, op, i32, int);

#define def_unary(opname, op)                       \
def_unary_kernels(opname, op, f32, float, float);   \
def_unary_kernels(opname, op, i32, int, int);

#define def_logic(opname, op)                       \
def_logic_kernels(opname, op);

#define def_bitwise(opname, op)                     \
def_unary_kernels(opname, op, i32, int, int);

def_unary_float(exp, Exp);
def_unary_float(log, Log);
def_unary(neg, Neg);
def_logic(logic_not, LogicNot);
def_bitwise(bitwise_not, BitwiseNot);
def_unary_float(recip, Recip);
def_unary_float(sin, Sin);
def_unary_float(cos, Cos);
def_unary(sq, Sq);
def_unary_float(sqrt, Sqrt);
