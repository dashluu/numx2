#pragma once

#include "array_buffer.h"
#include "device.h"
#include "dtype.h"
#include "shape.h"

namespace nx::foundation {
    struct ArrayDescriptor {
    private:
        Shape m_shape;
        const DType *m_dtype;
        const Device *m_device;
        bool m_is_param;
        ArrayBuffer *m_buff = nullptr;

    public:
        ArrayDescriptor(const Shape &shape, const DType *dtype, const Device *device, bool is_param) : m_shape(shape), m_dtype(dtype), m_device(device), m_is_param(is_param) {}

        ArrayDescriptor(std::uint8_t *ptr, usize nbytes, const Shape &shape, const DType *dtype, const Device *device, bool is_param) : ArrayDescriptor(shape, dtype, device, is_param) {
            m_buff = new ArrayBufferBorrower(new Buffer(ptr, nbytes));
        }

        ArrayDescriptor(const ArrayDescriptor &descriptor);
        ArrayDescriptor(ArrayDescriptor &&descriptor) noexcept;
        ~ArrayDescriptor() { invalidate_buffer(); }
        ArrayDescriptor &operator=(const ArrayDescriptor &descriptor);
        ArrayDescriptor &operator=(ArrayDescriptor &&descriptor) noexcept;
        const Shape &shape() const { return m_shape; }
        const ShapeView &view() const { return m_shape.get_view(); }
        const ShapeStride &stride() const { return m_shape.get_stride(); }
        const DType *dtype() const { return m_dtype; }
        const Device *device() const { return m_device; }
        const ArrayBuffer &buffer() const { return *m_buff; }
        bool is_buffer_valid() const { return m_buff != nullptr; }
        bool is_param() const { return m_is_param; }
        std::string device_name() const { return m_device->str(); }
        usize offset() const { return m_shape.get_offset(); }
        std::uint8_t *ptr() const { return m_buff->ptr() + offset() * itemsize(); }
        usize numel() const { return m_shape.get_numel(); }
        usize ndim() const { return m_shape.get_ndim(); }
        usize itemsize() const { return m_dtype->size(); }
        usize nbytes() const { return numel() * itemsize(); }
        usize size(usize dim) const { return m_shape.get_size(dim); }
        bool is_contiguous() const { return m_shape.is_contiguous(); }
        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) const { return !is_contiguous(); }

        void invalidate_buffer() {
            delete m_buff;
            m_buff = nullptr;
        }

        void alloc_buffer(BufferMemory *memory);
        void share_buffer(const ArrayDescriptor &descriptor);
        std::string str() const;
    };

    std::uint8_t *item_ptr(const ArrayDescriptor &descriptor, usize index);
} // namespace nx::foundation
