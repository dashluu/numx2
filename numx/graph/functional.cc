#include "functional.h"

namespace nx::graph {
    usize item(OpPtr op) {
        const ArrayDescriptor &descriptor = op->descriptor();

        if (descriptor.numel() != 1) {
            throw std::runtime_error(std::format("array must have exactly one element but has {} elements.", descriptor.numel()));
        }

        ArrayIterator iter(descriptor);
        iter.begin();
        std::uint8_t *ptr = iter.next();
        return descriptor.dtype()->bit_cast(ptr);
    }

    OpPtr gemm(OpPtr l_op, OpPtr r_op) {
        const ArrayDescriptor &l_descriptor = l_op->descriptor();
        const ArrayDescriptor &r_descriptor = r_op->descriptor();
        const ShapeView &l_view = l_descriptor.view();
        const ShapeView &r_view = r_descriptor.view();
        const DType *l_dtype = l_descriptor.dtype();
        const DType *r_dtype = r_descriptor.dtype();
        const Device *l_device = l_descriptor.device();
        const Device *r_device = r_descriptor.device();

        if (!l_descriptor.shape().gemm_broadcastable(r_view)) {
            throw foundation::IncompatShapesForOp(GemmOp::s_opname, foundation::join_nums(l_view), foundation::join_nums(r_view));
        }

        if (!foundation::is_numeric(l_dtype) || *l_dtype != *r_dtype) {
            throw foundation::IncompatDTypesForOp(GemmOp::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw foundation::IncompatDevicesForOp(GemmOp::s_opname, l_device->str(), r_device->str());
        }

        std::size_t l_ndim = l_view.size();
        std::size_t r_ndim = r_view.size();

        if (l_ndim == 1) {
            if (r_ndim == 1) {
                // gevv or vector dot product
                ArrayDescriptor gevv_descriptor(Shape({1}), l_dtype, l_device, false);
                return make_primitive<GemmOp>(gevv_descriptor, l_op, r_op);
            }
            // else {
            //     //  TODO: gevm
            // }
        }
        // else if (r_ndim == 1) {
        //     // TODO: gemv
        // }

        ShapeView broadcast_l_view = l_view;
        ShapeView broadcast_r_view = r_view;
        usize broadcast_ndim = std::max(l_ndim, r_ndim);
        broadcast_l_view.insert(broadcast_l_view.begin(), broadcast_ndim - broadcast_l_view.size(), 1);
        broadcast_r_view.insert(broadcast_r_view.begin(), broadcast_ndim - broadcast_r_view.size(), 1);

        for (usize i = 0; i < broadcast_ndim - 2; ++i) {
            usize shared_dim = std::max(broadcast_l_view[i], broadcast_r_view[i]);
            broadcast_l_view[i] = shared_dim;
            broadcast_r_view[i] = shared_dim;
        }

        OpPtr broadcast_l_op = broadcast(l_op, broadcast_l_view);
        OpPtr broadcast_r_op = broadcast(r_op, broadcast_r_view);
        ShapeView gemm_view = broadcast_l_view;
        gemm_view[gemm_view.size() - 1] = r_view[r_view.size() - 1];
        ArrayDescriptor gemm_descriptor(Shape(gemm_view), l_dtype, l_device, false);
        return make_primitive<GemmOp>(gemm_descriptor, broadcast_l_op, broadcast_r_op);
    }

    OpPtr copy(OpPtr in_op) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        ArrayDescriptor out_descriptor(Shape(in_descriptor.view()), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<CopyOp>(out_descriptor, in_op);
    }

    OpPtr broadcast(OpPtr in_op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const Shape &in_shape = in_descriptor.shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return in_op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast(view);

        if (in_shape == broadcast_shape) {
            return in_op;
        }

        ArrayDescriptor out_descriptor(broadcast_shape, in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<BroadcastOp>(out_descriptor, in_op, in_view, broadcast_dims);
    }

    OpPtr broadcast_to(OpPtr in_op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const Shape &in_shape = in_descriptor.shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return in_op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast_to(view);

        if (in_shape == broadcast_shape) {
            return in_op;
        }

        ArrayDescriptor out_descriptor(broadcast_shape, in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<BroadcastOp>(out_descriptor, in_op, in_view, broadcast_dims);
    }

    OpPtr astype(OpPtr in_op, const DType *dtype) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();

        if (in_descriptor.dtype() == dtype) {
            return in_op;
        }

        ArrayDescriptor out_descriptor(in_descriptor.shape(), dtype, in_descriptor.device(), false);
        return make_primitive<AstypeOp>(out_descriptor, in_op, dtype);
    }

    OpPtr slice(OpPtr in_op, const RangeVec &ranges) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        ArrayDescriptor out_descriptor(in_descriptor.shape().slice(ranges), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<SliceOp>(out_descriptor, in_op, ranges);
    }

    OpPtr reshape(OpPtr in_op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();

        if (in_descriptor.view() == view) {
            return in_op;
        }

        ArrayDescriptor out_descriptor(in_descriptor.shape().reshape(view), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<ReshapeOp>(out_descriptor, in_op);
    }

    OpPtr permute(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        ArrayDescriptor out_descriptor(in_descriptor.shape().permute(dims), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<PermuteOp>(out_descriptor, in_op, dims);
    }

    OpPtr transpose(OpPtr in_op, usize start_dim, usize end_dim) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ShapeDims &transpose_dims = in_descriptor.shape().transpose(start_dim, end_dim);
        return permute(in_op, transpose_dims);
    }

    OpPtr flatten(OpPtr in_op, usize start_dim, usize end_dim) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        const ShapeView &flattened_view = in_descriptor.shape().flatten(start_dim, end_dim);
        return reshape(in_op, flattened_view);
    }

    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        ArrayDescriptor out_descriptor(in_descriptor.shape().unsqueeze(dims), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<UnsqueezeOp>(out_descriptor, in_op, dims);
    }

    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->descriptor();
        ArrayDescriptor out_descriptor(in_descriptor.shape().squeeze(dims), in_descriptor.dtype(), in_descriptor.device(), false);
        return make_primitive<SqueezeOp>(out_descriptor, in_op, dims);
    }

    OpPtr mean(OpPtr in_op, const ShapeDims &dims) {
        OpPtr sum_op = sum(in_op, dims);
        usize numel;

        if (dims.empty()) {
            numel = in_op->descriptor().numel();
        } else {
            const ShapeView &in_view = in_op->descriptor().view();
            numel = std::accumulate(dims.begin(), dims.end(), uone, [&](usize acc, usize dim) { return acc * in_view[dim]; });
        }

        return div(sum_op, numel);
    }

    OpPtr expand(OpPtr in_op, const ShapeView &reduce_operand_view, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) {
        // TODO: check if remaining_dims and reduce_dims are valid?
        usize reduce_numel = std::accumulate(reduce_dims.begin(), reduce_dims.end(), uone, [&](usize acc, usize dim) { return acc * reduce_operand_view[dim]; });

        // Broadcast the last dimension so the last dimension is the number of reduced elements
        ShapeView broadcast_view(remaining_dims.size() + 1, reduce_numel);
        std::transform(remaining_dims.begin(), remaining_dims.end(), broadcast_view.begin(), [&](usize dim) { return reduce_operand_view[dim]; });
        OpPtr out_op = broadcast(in_op, broadcast_view);

        // Reshape so the last k dimensions are the reduced dimensions
        ShapeView reshape_view(remaining_dims.size() + reduce_dims.size());
        std::transform(remaining_dims.begin(), remaining_dims.end(), reshape_view.begin(), [&](usize dim) { return reduce_operand_view[dim]; });
        std::transform(reduce_dims.begin(), reduce_dims.end(), reshape_view.begin() + remaining_dims.size(), [&](usize dim) { return reduce_operand_view[dim]; });
        out_op = reshape(out_op, reshape_view);

        // Permute to restore the shape before reduction
        ShapeDims permute_dims;
        permute_dims.reserve(remaining_dims.size() + reduce_dims.size());
        permute_dims.insert(permute_dims.end(), remaining_dims.begin(), remaining_dims.end());
        permute_dims.insert(permute_dims.end(), reduce_dims.begin(), reduce_dims.end());
        ShapeDims pre_permute_dims = out_op->descriptor().shape().undo_permute_dims(permute_dims);
        out_op = permute(out_op, pre_permute_dims);
        return out_op;
    }
} // namespace nx::graph
