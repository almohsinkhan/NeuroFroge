#include "flatten.h"

#include <cassert>

int main() {
    Flatten flatten;

    Tensor input({2, 2, 2, 2});
    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = i + 1;
    }

    Tensor output = flatten.forward(input);

    assert(output.getShape() == std::vector<int>({8, 2}));

    for (int feature = 0; feature < 8; feature++) {
        assert(output.flat(feature * 2) == feature + 1);
        assert(output.flat(feature * 2 + 1) == feature + 9);
    }

    Tensor restored = flatten.backward(input, output);

    assert(restored.getShape() == std::vector<int>({2, 2, 2, 2}));
    for (int i = 0; i < restored.size(); i++) {
        assert(restored.flat(i) == input.flat(i));
    }
}
