#include "execution.h"
#include "maxpool2d.h"

#include <cassert>
#include <cmath>

void assertEqual(
    const Tensor& left,
    const Tensor& right
) {
    assert(left.getShape() == right.getShape());
    for (int i = 0; i < left.size(); i++) {
        assert(std::abs(left.flat(i) - right.flat(i)) < 1e-12);
    }
}

Tensor run(
    const Tensor& input,
    int kernel_height,
    int kernel_width,
    int stride,
    std::size_t threshold
) {
    execution::setParallelThreshold(threshold);
    MaxPool2D pool(kernel_height, kernel_width, stride);
    return pool.forward(input);
}

int main() {
    Tensor non_overlapping({2, 2, 4, 4});
    for (int i = 0; i < non_overlapping.size(); i++) {
        non_overlapping.flat(i) = i + 1;
    }

    Tensor serial = run(non_overlapping, 2, 2, 2, 1000000);
    Tensor parallel = run(non_overlapping, 2, 2, 2, 1);
    assertEqual(serial, parallel);

    Tensor overlapping({2, 2, 4, 4});
    for (int i = 0; i < overlapping.size(); i++) {
        overlapping.flat(i) = (i % 7) * 0.25;
    }

    serial = run(overlapping, 2, 2, 1, 1000000);
    parallel = run(overlapping, 2, 2, 1, 1);
    assertEqual(serial, parallel);

    Tensor tied({1, 1, 3, 3});
    for (int i = 0; i < tied.size(); i++) {
        tied.flat(i) = 1.0;
    }

    serial = run(tied, 2, 2, 1, 1000000);
    parallel = run(tied, 2, 2, 1, 1);
    assertEqual(serial, parallel);

    Tensor base({1, 2, 2, 2});
    for (int i = 0; i < base.size(); i++) {
        base.flat(i) = i + 1;
    }

    Tensor view = base.broadcastTo({2, 2, 2, 2});
    serial = run(view, 2, 2, 1, 1000000);
    parallel = run(view, 2, 2, 1, 1);
    assertEqual(serial, parallel);
}
