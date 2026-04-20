#pragma once

#include "array.h"
#include "functional.h"

namespace nx::bind {
    inline Array onehot(const Array &x, isize num_classes) { return nxn::onehot(x, num_classes > 0 ? num_classes : x.max().item() + 1); }
    inline Array softmax(const Array &x, isize dim) { return nxn::softmax(x, py_index(x.ndim(), dim)); }
} // namespace nx::bind
