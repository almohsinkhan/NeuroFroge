#include "execution.h"
#include "maxpool2d.h"

#include <cassert>
#include <cmath>

void assertClose(
    const Tensor& left,
    const Tensor& right
) {
    assert(left.getShape() == right.getShape());
    for (int i = 0; i < left.size(); i++) {
        assert(std::abs(left.flat(i) - right.flat(i)) < 1e-12);
    }
}

Tensor runBackward(
    const Tensor& input,
    const Tensor& dZ,
    int kernel_height,
    int kernel_width,
    int stride,
    std::size_t threshold
) {
    execution::setParallelThreshold(threshold);
    MaxPool2D pool(kernel_height, kernel_width, stride);
    pool.forward(input);
    return pool.backward(input, dZ);
}

int main() {
    Tensor input({2, 2, 4, 4});
    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = 0.2 * (i + 1);
    }

    Tensor non_overlapping_dZ({2, 2, 2, 2});
    for (int i = 0; i < non_overlapping_dZ.size(); i++) {
        non_overlapping_dZ.flat(i) = 0.1 * (i + 1);
    }

    Tensor serial = runBackward(
        input,
        non_overlapping_dZ,
        2,
        2,
        2,
        1000000
    );
    Tensor parallel = runBackward(
        input,
        non_overlapping_dZ,
        2,
        2,
        2,
        1
    );
    assertClose(serial, parallel);

    Tensor overlapping_dZ({2, 2, 3, 3});
    for (int i = 0; i < overlapping_dZ.size(); i++) {
        overlapping_dZ.flat(i) = 0.05 * (i + 1);
    }

    serial = runBackward(
        input,
        overlapping_dZ,
        2,
        2,
        1,
        1000000
    );
    parallel = runBackward(
        input,
        overlapping_dZ,
        2,
        2,
        1,
        1
    );
    assertClose(serial, parallel);
}
