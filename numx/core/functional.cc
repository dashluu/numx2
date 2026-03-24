#include "functional.h"

namespace nx::core {
    std::pair<usize, usize> compute_fan_in_and_fan_out(const ShapeView &view) {
        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for a view of {} dimensions, which is fewer than 2.", view.size()));
        }

        usize receptive_field_size = std::accumulate(view.begin() + 2, view.end(), uone, std::multiplies<usize>());
        usize fan_in = view[1] * receptive_field_size;
        usize fan_out = view[0] * receptive_field_size;
        // Note: fan-in and fan-out cannot be 0 since each dimension > 0
        return {fan_in, fan_out};
    }

    std::pair<usize, usize> compute_fan_in_and_fan_out(const Array &array) {
        const ShapeView &view = array.view();

        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for array of {} dimensions, which is fewer than 2.", view.size()));
        }

        return compute_fan_in_and_fan_out(view);
    }
} // namespace nx::core
