#include "maxpool2d.h"

#include <cassert>

int main() {
    MaxPool2D pool(2, 2, 2);

    Tensor input({1, 1, 4, 4});
    const double values[] = {
        1, 3, 2, 4,
        5, 9, 6, 2,
        7, 1, 8, 3,
        4, 2, 5, 6
    };

    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = values[i];
    }

    Tensor output = pool.forward(input);

    assert(output.getShape() == std::vector<int>({1, 1, 2, 2}));

    const double expected[] = {9, 6, 7, 8};
    for (int i = 0; i < output.size(); i++) {
        assert(output.flat(i) == expected[i]);
    }
}
