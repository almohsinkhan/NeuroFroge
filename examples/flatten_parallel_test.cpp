#include "execution.h"
#include "flatten.h"

#include <cassert>
#include <cmath>

void assertEqual(
    const Tensor& left,
    const Tensor& right
) {
    assert(left.getShape() == right.getShape());
    assert(left.size() == right.size());

    for (int i = 0; i < left.size(); i++) {
        assert(std::abs(left.flat(i) - right.flat(i)) < 1e-12);
    }
}

int main() {
    Tensor base({1, 2, 2, 2});
    for (int i = 0; i < base.size(); i++) {
        base.flat(i) = i + 1;
    }

    Tensor input = base.broadcastTo({2, 2, 2, 2});
    Flatten serial_flatten;
    Flatten parallel_flatten;

    execution::setParallelThreshold(input.size() + 1);
    Tensor serial_output = serial_flatten.forward(input);
    Tensor serial_gradient = serial_flatten.backward(input, serial_output);

    execution::setParallelThreshold(1);
    Tensor parallel_output = parallel_flatten.forward(input);
    Tensor parallel_gradient =
        parallel_flatten.backward(input, parallel_output);

    assertEqual(serial_output, parallel_output);
    assertEqual(serial_gradient, parallel_gradient);
    for (int i = 0; i < input.size(); i++) {
        assert(serial_gradient.flat(i) == input.flat(i));
    }
}
