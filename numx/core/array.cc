#include "array.h"

namespace nx::core {
    Runtime *backend_runtime(usize device_id, DeviceKind device_kind) {
        return Backend::instance().runtime(device_id, device_kind);
    }

    Array::Array(OpPtr op) : m_op(op) {
        ArrayDescriptor &descriptor = m_op->descriptor();
        const Device *device = descriptor.device();
        m_runtime = backend_runtime(device->id(), device->kind());
    }

    Array::~Array() {
        if (m_graph) {
            // Free non-parameter buffers on the forward tape
            for (auto iter = m_graph->fw_begin(); iter != m_graph->fw_end(); ++iter) {
                Op *op = *iter;

                if (!op->is_param()) {
                    ArrayDescriptor &descriptor = op->descriptor();
                    descriptor.invalidate_buffer();
                }
            }

            // Free buffers on the backward tape
            // Free even if the buffers are parameters so the gradient is not affected if the computational graph changes
            for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
                ArrayDescriptor &descriptor = (*iter)->descriptor();
                descriptor.invalidate_buffer();
            }

            m_graph->clear_grad();
        }
    }

    void Array::eval() {
        if (!m_op->descriptor().is_buffer_valid()) {
            m_graph = graph::make_graph(m_op);
            m_graph->fw_sort();
            m_runtime->forward(m_graph.get());
        }
    }

    void Array::backward() {
        eval();
        m_graph->bw_sort();
        m_runtime->backward(m_graph.get());
    }
} // namespace nx::core
