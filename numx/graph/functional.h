#pragma once

#include "../foundation/array_iterator.h"
#include "../foundation/exceptions.h"
#include "../foundation/random.h"
#include "binary_op.h"
#include "initializer_op.h"
#include "reduce_op.h"
#include "transform_op.h"
#include "unary_op.h"
#include <numbers>

namespace nx::graph {
    using foundation::ArrayBuffer;
    using foundation::ArrayIterator;
    using foundation::Device;
    using foundation::NumericOrBoolType;
    using foundation::NumericType;
    using foundation::RandomKeyGenerator;
    using foundation::Shape;
    using foundation::uone;
    using foundation::usize;
    using memory::Buffer;

    // Extracts a scalar value from a single-element array
    usize item(OpPtr op);

    // Creates a node from raw buffer data
    inline OpPtr from_buffer(uint8_t *ptr, usize nbytes, const Shape &shape, const DType *dtype, const Device *device, bool is_param) {
        return make_primitive<Nop>(ArrayDescriptor(ptr, nbytes, shape, dtype, device, is_param));
    }

    // Creates a new array from an existing buffer, breaking the computational graph
    inline OpPtr detach(OpPtr op, bool is_param = false) {
        const ArrayDescriptor &descriptor = op->descriptor();
        const ArrayBuffer &buff = descriptor.buffer();
        return from_buffer(buff.ptr(), buff.nbytes(), descriptor.shape(), descriptor.dtype(), descriptor.device(), is_param);
    }

    inline OpPtr empty(const ShapeView &view, const DType *dtype, const Device *device, bool is_param = false) {
        return make_primitive<EmptyOp>(ArrayDescriptor(Shape(view), dtype, device, is_param));
    }

    inline OpPtr empty_like(OpPtr op, const DType *dtype, const Device *device, bool is_param = false) {
        return empty(op->descriptor().view(), dtype, device, is_param);
    }

    inline OpPtr empty_like(OpPtr op, bool is_param = false) {
        const ArrayDescriptor &descriptor = op->descriptor();
        return empty(descriptor.view(), descriptor.dtype(), descriptor.device(), is_param);
    }

    template <NumericOrBoolType T>
    OpPtr full(const ShapeView &view, T constant, const DType *dtype, const Device *device, bool is_param = false) {
        return make_primitive<FullOp>(ArrayDescriptor(Shape(view), dtype, device, is_param), numeric_bitcast(dtype, constant));
    }

    template <NumericOrBoolType T>
    OpPtr full_like(OpPtr in_op, T constant, const DType *dtype, const Device *device, bool is_param = false) {
        return full(in_op->descriptor().view(), constant, dtype, device, is_param);
    }

    template <NumericOrBoolType T>
    OpPtr full_like(OpPtr in_op, T constant, bool is_param = false) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        return full(in_descriptor.view(), constant, in_descriptor.dtype(), in_descriptor.device(), is_param);
    }

    // Creates array with values from start to end with given step
    inline OpPtr arange(const ShapeView &view, isize start, isize step, const DType *dtype, const Device *device, bool is_param = false) {
        return make_primitive<ArangeOp>(ArrayDescriptor(Shape(view), dtype, device, is_param), numeric_bitcast(dtype, start), numeric_bitcast(dtype, step));
    }

    template <NumericType T>
    OpPtr uniform(const ShapeView &view, RandomKeyGenerator *key_gen, T low, T high, const DType *dtype, const Device *device, bool is_param = false) {
        uint64_t key = key_gen->next();
        return make_primitive<UniformOp>(ArrayDescriptor(Shape(view), dtype, device, is_param), key, numeric_bitcast(dtype, low), numeric_bitcast(dtype, high));
    }

    inline OpPtr zeros(const ShapeView &view, const DType *dtype, const Device *device, bool is_param = false) { return full(view, 0, dtype, device, is_param); }
    inline OpPtr zeros_like(OpPtr in_op, const DType *dtype, const Device *device, bool is_param = false) { return full_like(in_op, 0, dtype, device, is_param); }
    inline OpPtr zeros_like(OpPtr in_op, bool is_param = false) { return full_like(in_op, 0, is_param); }
    inline OpPtr ones(const ShapeView &view, const DType *dtype, const Device *device, bool is_param = false) { return full(view, 1, dtype, device, is_param); }
    inline OpPtr ones_like(OpPtr in_op, const DType *dtype, const Device *device, bool is_param = false) { return full_like(in_op, 1, dtype, device, is_param); }
    inline OpPtr ones_like(OpPtr in_op, bool is_param = false) { return full_like(in_op, 1, is_param); }
    OpPtr broadcast(OpPtr in_op, const ShapeView &view);
    OpPtr broadcast_to(OpPtr in_op, const ShapeView &view);

    template <class O>
    OpPtr unary(OpPtr in_op, bool (*check_dtype)(const DType *), bool in_place) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const DType *in_dtype = in_descriptor.dtype();

        if (check_dtype && !check_dtype(in_dtype)) {
            throw foundation::IncompatDTypeForOp(O::s_opname, in_dtype->str());
        }

        ArrayDescriptor out_descriptor(Shape(in_descriptor.view()), in_dtype, in_descriptor.device(), false);
        return make_primitive<O>(out_descriptor, in_op, in_place);
    }

    template <class O>
    OpPtr float_unary(OpPtr in_op, bool in_place) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const DType *in_dtype = in_descriptor.dtype();

        if (in_place) {
            if (!is_float(in_dtype)) {
                // This method requires the operand to be of floating-point type
                // to do in-place operation since the result is of floating-point type
                throw foundation::IncompatDTypeForOp(O::s_opname, in_dtype->str());
            }
        }

        const DType *out_dtype = float_dtype_by_dtype(in_dtype);

        if (!out_dtype) {
            throw foundation::IncompatDTypeForOp(O::s_opname, in_dtype->str());
        }

        ArrayDescriptor out_descriptor(Shape(in_descriptor.view()), out_dtype, in_descriptor.device(), false);
        return make_primitive<O>(out_descriptor, in_op, in_place);
    }

    // Creates a deep copy of the array
    OpPtr copy(OpPtr in_op);
    inline OpPtr sq(OpPtr in_op, bool in_place = false) { return unary<SqOp>(in_op, foundation::is_numeric, in_place); }
    inline OpPtr neg(OpPtr in_op, bool in_place = false) { return unary<NegOp>(in_op, foundation::is_numeric, in_place); }
    inline OpPtr logic_not(OpPtr in_op, bool in_place = false) { return unary<LogicNot>(in_op, foundation::is_bool, in_place); }
    inline OpPtr bitwise_not(OpPtr in_op, bool in_place = false) { return unary<BitwiseNot>(in_op, foundation::is_int, in_place); }
    inline OpPtr sqrt(OpPtr in_op, bool in_place = false) { return float_unary<SqrtOp>(in_op, in_place); }
    inline OpPtr exp(OpPtr in_op, bool in_place = false) { return float_unary<ExpOp>(in_op, in_place); }
    inline OpPtr log(OpPtr in_op, bool in_place = false) { return float_unary<LogOp>(in_op, in_place); }
    inline OpPtr recip(OpPtr in_op, bool in_place = false) { return float_unary<RecipOp>(in_op, in_place); }
    inline OpPtr sin(OpPtr in_op, bool in_place = false) { return float_unary<SinOp>(in_op, in_place); }
    inline OpPtr cos(OpPtr in_op, bool in_place = false) { return float_unary<CosOp>(in_op, in_place); }

    template <NumericType T>
    OpPtr binary_with_scalar(OpPtr l_op, T constant, OpPtr (*op_fn)(OpPtr, OpPtr)) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const DType *l_dtype = l_descriptor.dtype();
        OpPtr r_op = full(l_descriptor.view(), constant, l_dtype, l_descriptor.device());
        r_op->enable_grad(false);
        return op_fn(l_op, r_op);
    }

    template <NumericOrBoolType T>
    OpPtr eq_with_scalar(OpPtr l_op, T constant, OpPtr (*op_fn)(OpPtr, OpPtr)) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const DType *l_dtype = l_descriptor.dtype();
        OpPtr r_op = full(l_descriptor.view(), constant, l_dtype, l_descriptor.device());
        r_op->enable_grad(false);
        return op_fn(l_op, r_op);
    }

    template <class O>
    OpPtr binary(OpPtr l_op, OpPtr r_op, bool (*check_dtype)(const DType *)) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ShapeView &l_view = l_descriptor.view();
        const ShapeView &r_view = r_descriptor.view();
        const DType *l_dtype = l_descriptor.dtype();
        const DType *r_dtype = r_descriptor.dtype();
        const Device *l_device = l_descriptor.device();
        const Device *r_device = r_descriptor.device();

        if (!l_descriptor.shape().broadcastable(r_view)) {
            throw foundation::IncompatShapesForOp(O::s_opname, foundation::join_nums(l_view), foundation::join_nums(r_view));
        }

        if ((check_dtype && !check_dtype(l_dtype)) || *l_dtype != *r_dtype) {
            throw foundation::IncompatDTypesForOp(O::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw foundation::IncompatDevicesForOp(O::s_opname, l_device->str(), r_device->str());
        }

        OpPtr broadcast_l_op = broadcast(l_op, r_view);
        OpPtr broadcast_r_op = broadcast(r_op, l_view);
        ArrayDescriptor out_descriptor(Shape(broadcast_l_op->descriptor().view()), l_dtype, l_device, false);
        return make_primitive<O>(out_descriptor, broadcast_l_op, broadcast_r_op, false);
    }

    template <class O>
    OpPtr in_place_binary(OpPtr l_op, OpPtr r_op, bool (*check_dtype)(const DType *)) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const Shape &l_shape = l_descriptor.shape();
        const ShapeView &l_view = l_descriptor.view();
        const ShapeView &r_view = r_descriptor.view();
        const DType *l_dtype = l_descriptor.dtype();
        const DType *r_dtype = r_descriptor.dtype();
        const Device *l_device = l_descriptor.device();
        const Device *r_device = r_descriptor.device();

        if (!l_descriptor.shape().broadcastable(r_view)) {
            throw foundation::IncompatShapesForOp(O::s_opname, foundation::join_nums(l_view), foundation::join_nums(r_view));
        }

        if ((check_dtype && !check_dtype(l_dtype)) || *l_dtype != *r_dtype) {
            throw foundation::IncompatDTypesForOp(O::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw foundation::IncompatDevicesForOp(O::s_opname, l_device->str(), r_device->str());
        }

        OpPtr broadcast_r_op = broadcast_to(r_op, l_view);
        ArrayDescriptor out_descriptor(l_shape, l_dtype, l_device, false);
        return make_primitive<O>(out_descriptor, l_op, broadcast_r_op, true);
    }

    // Element-wise binary operations
    inline OpPtr add(OpPtr l_op, OpPtr r_op) { return binary<AddOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_add(OpPtr l_op, OpPtr r_op) { return in_place_binary<AddOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr sub(OpPtr l_op, OpPtr r_op) { return binary<SubOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_sub(OpPtr l_op, OpPtr r_op) { return in_place_binary<SubOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr mul(OpPtr l_op, OpPtr r_op) { return binary<MulOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_mul(OpPtr l_op, OpPtr r_op) { return in_place_binary<MulOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr div(OpPtr l_op, OpPtr r_op) { return binary<DivOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_div(OpPtr l_op, OpPtr r_op) { return in_place_binary<DivOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr logic_and(OpPtr l_op, OpPtr r_op) { return binary<LogicAndOp>(l_op, r_op, foundation::is_bool); }
    inline OpPtr i_logic_and(OpPtr l_op, OpPtr r_op) { return in_place_binary<LogicAndOp>(l_op, r_op, foundation::is_bool); }
    inline OpPtr logic_or(OpPtr l_op, OpPtr r_op) { return binary<LogicOrOp>(l_op, r_op, foundation::is_bool); }
    inline OpPtr i_logic_or(OpPtr l_op, OpPtr r_op) { return in_place_binary<LogicOrOp>(l_op, r_op, foundation::is_bool); }
    inline OpPtr bitwise_and(OpPtr l_op, OpPtr r_op) { return binary<BitwiseAndOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr i_bitwise_and(OpPtr l_op, OpPtr r_op) { return in_place_binary<BitwiseAndOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr bitwise_or(OpPtr l_op, OpPtr r_op) { return binary<BitwiseOrOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr i_bitwise_or(OpPtr l_op, OpPtr r_op) { return in_place_binary<BitwiseOrOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr bitwise_xor(OpPtr l_op, OpPtr r_op) { return binary<BitwiseXorOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr i_bitwise_xor(OpPtr l_op, OpPtr r_op) { return in_place_binary<BitwiseXorOp>(l_op, r_op, foundation::is_int); }
    inline OpPtr minimum(OpPtr l_op, OpPtr r_op) { return binary<MinimumOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_minimum(OpPtr l_op, OpPtr r_op) { return in_place_binary<MinimumOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr maximum(OpPtr l_op, OpPtr r_op) { return binary<MaximumOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr i_maximum(OpPtr l_op, OpPtr r_op) { return in_place_binary<MaximumOp>(l_op, r_op, foundation::is_numeric); }
    // Matrix multiplication with automatic broadcasting for batch dimensions
    OpPtr gemm(OpPtr l_op, OpPtr r_op);

    // Comparison operations (eq, neq, less, greater, leq, geq)
    inline OpPtr eq(OpPtr l_op, OpPtr r_op) { return binary<EqOp>(l_op, r_op, nullptr); }
    inline OpPtr neq(OpPtr l_op, OpPtr r_op) { return binary<NeqOp>(l_op, r_op, nullptr); }
    inline OpPtr less(OpPtr l_op, OpPtr r_op) { return binary<LessOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr greater(OpPtr l_op, OpPtr r_op) { return binary<GreaterOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr leq(OpPtr l_op, OpPtr r_op) { return binary<LessOp>(l_op, r_op, foundation::is_numeric); }
    inline OpPtr geq(OpPtr l_op, OpPtr r_op) { return binary<GreaterOp>(l_op, r_op, foundation::is_numeric); }

    template <NumericType T>
    OpPtr add(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, add); }

    template <NumericType T>
    OpPtr i_add(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_add); }

    template <NumericType T>
    OpPtr sub(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, sub); }

    template <NumericType T>
    OpPtr i_sub(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_sub); }

    template <NumericType T>
    OpPtr mul(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, mul); }

    template <NumericType T>
    OpPtr i_mul(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_mul); }

    template <NumericType T>
    OpPtr div(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, div); }

    template <NumericType T>
    OpPtr i_div(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_div); }

    template <NumericType T>
    OpPtr bitwise_and(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, bitwise_and); }

    template <NumericType T>
    OpPtr i_bitwise_and(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_bitwise_and); }

    template <NumericType T>
    OpPtr logic_and(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, logic_and); }

    template <NumericType T>
    OpPtr i_logic_and(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_logic_and); }

    template <NumericType T>
    OpPtr bitwise_or(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, bitwise_or); }

    template <NumericType T>
    OpPtr i_bitwise_or(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_bitwise_or); }

    template <NumericType T>
    OpPtr logic_or(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, logic_or); }

    template <NumericType T>
    OpPtr i_logic_or(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_logic_or); }

    template <NumericType T>
    OpPtr bitwise_xor(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, bitwise_xor); }

    template <NumericType T>
    OpPtr i_bitwise_xor(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, i_bitwise_xor); }

    template <NumericOrBoolType T>
    OpPtr eq(OpPtr l_op, T constant) { return eq_with_scalar(l_op, constant, eq); }

    template <NumericOrBoolType T>
    OpPtr neq(OpPtr l_op, T constant) { return eq_with_scalar(l_op, constant, neq); }

    template <NumericType T>
    OpPtr less(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, less); }

    template <NumericType T>
    OpPtr greater(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, greater); }

    template <NumericType T>
    OpPtr minimum(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, minimum); }

    template <NumericType T>
    OpPtr maximum(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, maximum); }

    template <NumericType T>
    OpPtr normal(const ShapeView &view, RandomKeyGenerator *key_gen, T mean, T std, const DType *dtype, const Device *device, bool is_param = false) {
        // TODO: cache second output by Box-Muller transform for future use?
        OpPtr lhs = uniform(view, key_gen, 0, 1, dtype, device, is_param);
        OpPtr rhs = uniform(view, key_gen, 0, 1, dtype, device, is_param);
        OpPtr std_normal = mul(sqrt(mul(log(lhs), -2)), cos(mul(rhs, 2 * std::numbers::pi)));
        return add(mul(std_normal, std), mean);
    }

    // Converts array to different data type
    OpPtr astype(OpPtr in_op, const DType *dtype);
    // Extracts a contiguous subarray based on ranges
    OpPtr slice(OpPtr in_op, const RangeVec &ranges);
    OpPtr reshape(OpPtr in_op, const ShapeView &view);
    // Reorders dimensions according to the specified permutation
    OpPtr permute(OpPtr in_op, const ShapeDims &dims);
    // Transposes the array by swapping two dimensions
    OpPtr transpose(OpPtr in_op, usize start_dim, usize end_dim);
    // Collapses a range of dimensions into a single dimension
    OpPtr flatten(OpPtr in_op, usize start_dim, usize end_dim);
    // Adds dimensions of size 1 at specified positions
    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims);
    // Removes dimensions of size 1 at specified positions
    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims);

    template <class O>
    OpPtr reduce(OpPtr in_op, const ShapeDims &dims, const DType *out_dtype, bool (*check_dtype)(const DType *)) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const Shape &in_shape = in_descriptor.shape();
        const DType *in_dtype = in_descriptor.dtype();
        const Device *in_device = in_descriptor.device();

        if (check_dtype && !check_dtype(in_dtype)) {
            throw foundation::IncompatDTypeForOp(O::s_opname, in_dtype->str());
        }

        OpPtr out_op;
        ShapeDims reduce_dims;
        ShapeDims remaining_dims(in_shape.get_ndim());
        std::iota(remaining_dims.begin(), remaining_dims.end(), 0);

        if (dims.size() == 0) {
            // Reduce to one element
            reduce_dims = remaining_dims;
            remaining_dims.clear();
            ArrayDescriptor out_descriptor(Shape({1}), out_dtype, in_device, false);
            out_op = make_primitive<O>(out_descriptor, in_op, remaining_dims, reduce_dims);
            return out_op;
        }

        // Remove the dimensions to be reduced from remaining dims
        for (auto &dim : dims) {
            auto iter = std::find(remaining_dims.begin(), remaining_dims.end(), dim);

            if (iter == remaining_dims.end()) {
                throw std::invalid_argument(std::format("invalid reduction dimension {} on array, either it does not exist or is duplicated.", dim));
            } else {
                remaining_dims.erase(iter);
                reduce_dims.push_back(dim);
            }
        }

        ShapeView out_view(remaining_dims.size() + 1, 1);
        std::transform(remaining_dims.begin(), remaining_dims.end(), out_view.begin(), [&](usize dim) { return in_shape[dim]; });
        ArrayDescriptor out_descriptor(Shape(out_view), out_dtype, in_device, false);
        out_op = make_primitive<O>(out_descriptor, in_op, remaining_dims, reduce_dims);
        return out_op;
    }

    // Computes the mean by summing and dividing by the number of reduced elements
    OpPtr mean(OpPtr in_op, const ShapeDims &dims = {});
    inline OpPtr sum(OpPtr in_op, const ShapeDims &dims = {}) { return reduce<SumOp>(in_op, dims, in_op->descriptor().dtype(), foundation::is_numeric); }
    inline OpPtr max(OpPtr in_op, const ShapeDims &dims = {}) { return reduce<MaxOp>(in_op, dims, in_op->descriptor().dtype(), foundation::is_numeric); }
    inline OpPtr min(OpPtr in_op, const ShapeDims &dims = {}) { return reduce<MinOp>(in_op, dims, in_op->descriptor().dtype(), foundation::is_numeric); }
    inline OpPtr argmax(OpPtr in_op, const ShapeDims &dims = {}) { return reduce<ArgmaxOp>(in_op, dims, &foundation::i32, foundation::is_numeric); }
    inline OpPtr argmin(OpPtr in_op, const ShapeDims &dims = {}) { return reduce<ArgminOp>(in_op, dims, &foundation::i32, foundation::is_numeric); }
    // Expands reduced dimensions back to their original size for gradient computation
    OpPtr expand(OpPtr in_op, const ShapeView &reduce_operand_view, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims);
} // namespace nx::graph
