#include "runtime.h"

namespace nx::runtime {
    void Runtime::run_op(Op *op) {
        switch (op->kind()) {
        case PrimitiveKind::Initializer: {
            run_initializer_op(op);
            break;
        }
        case PrimitiveKind::Unary: {
            run_unary_op(op);
            break;
        }
        case PrimitiveKind::Binary: {
            run_binary_op(op);
            break;
        }
        case PrimitiveKind::Transform: {
            run_transform_op(op);
            break;
        }
        default: {
            run_reduce_op(op);
            break;
        }
        }
    }

    void Runtime::forward(Graph *graph) {
        for (auto iter = graph->fw_begin(); iter != graph->fw_end(); ++iter) {
            run_op(*iter);
        }
    }

    void Runtime::backward(Graph *graph) {
        for (auto iter = graph->bw_begin(); iter != graph->bw_end(); ++iter) {
            run_op(*iter);
        }
    }
} // namespace nx::runtime
