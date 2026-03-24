#include "array_descriptor.h"
#include "array_iterator.h"

namespace nx::foundation {
    ArrayDescriptor::ArrayDescriptor(const ArrayDescriptor &descriptor) {
        m_shape = descriptor.m_shape;
        m_dtype = descriptor.m_dtype;
        m_device = descriptor.m_device;
        m_is_param = false;

        if (descriptor.m_buff) {
            m_buff = new ArrayBufferBorrower(*descriptor.m_buff);
        }
    }

    ArrayDescriptor &ArrayDescriptor::operator=(const ArrayDescriptor &descriptor) {
        delete m_buff;
        m_shape = descriptor.m_shape;
        m_dtype = descriptor.m_dtype;
        m_device = descriptor.m_device;
        m_is_param = false;

        if (descriptor.m_buff) {
            m_buff = new ArrayBufferBorrower(*descriptor.m_buff);
        }
    }

    void ArrayDescriptor::alloc_buffer(BufferMemory *memory) {
        if (!m_buff) {
            Buffer *allocated = memory->alloc(nbytes());
            m_buff = new ArrayBufferOwner(allocated, memory);
        }
    }

    void ArrayDescriptor::share_buffer(const ArrayDescriptor &descriptor) {
        if (!m_buff) {
            m_buff = new ArrayBufferBorrower(*descriptor.m_buff);
        }
    }

    std::string ArrayDescriptor::str() const {
        ArrayIterator iter(*this);
        iter.begin();
        bool next_available = iter.has_next();

        if (!next_available) {
            return "[]";
        }

        std::string s = "";

        for (usize i = 0; i < ndim(); ++i) {
            s += "[";
        }

        ShapeView size_per_dim = m_shape.size_per_dim();
        std::size_t close = 0;

        while (next_available) {
            close = 0;
            std::uint8_t *ptr = iter.next();
            s += m_dtype->value(ptr);

            for (std::ptrdiff_t i = size_per_dim.size() - 1; i >= 0; --i) {
                if (iter.count() % size_per_dim[i] == 0) {
                    s += "]";
                    close += 1;
                }
            }

            next_available = iter.has_next();

            if (next_available) {
                if (close > 0) {
                    s += ", \n";
                } else {
                    s += ", ";
                }

                for (std::size_t i = 0; i < close; ++i) {
                    s += "[";
                }
            }
        }

        return s;
    }

    std::uint8_t *item_ptr(const ArrayDescriptor &descriptor, usize index) {
        if (descriptor.is_contiguous()) {
            return descriptor.ptr() + index * descriptor.itemsize();
        }

        const ShapeView &view = descriptor.view();
        const ShapeStride &stride = descriptor.stride();
        std::uint8_t *ptr = descriptor.ptr();
        usize carry = index;

        for (isize i = descriptor.ndim() - 1; i >= 0; --i) {
            ptr += (carry % view[i]) * stride[i] * descriptor.itemsize();
            carry /= view[i];
        }

        return ptr;
    }
} // namespace nx::foundation
