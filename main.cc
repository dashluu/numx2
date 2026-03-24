#include "numx/core/functional.h"

using namespace nx::core;

int main() {
    auto x1 = full({1, 2, 3}, 1);
    auto x2 = full({1, 2, 3}, 2);
    std::cout << x1.str() << std::endl;
    std::cout << x2.str() << std::endl;
    auto x3 = x1 + x2;
    std::cout << x3.str() << std::endl;
    auto x4 = x2 * x2;
    std::cout << x4.str() << std::endl;
    return 0;
}
