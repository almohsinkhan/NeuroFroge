#include "execution.h"
#include "relu.h"

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
    Tensor base({1, 4});
    base.flat(0) = -2.0;
    base.flat(1) = 0.0;
    base.flat(2) = 3.0;
    base.flat(3) = -4.0;

    Tensor input = base.broadcastTo({3, 4});
    Tensor grad_output({3, 4});

    for (int i = 0; i < grad_output.size(); i++) {
        grad_output.flat(i) = 0.5 * (i + 1);
    }

    ReLU relu_layer;

    execution::setParallelThreshold(input.size() + 1);
    Tensor serial_forward = relu_layer.forward(input);
    Tensor serial_backward = relu_layer.backward(input, grad_output);

    execution::setParallelThreshold(1);
    Tensor parallel_forward = relu_layer.forward(input);
    Tensor parallel_backward = relu_layer.backward(input, grad_output);

    assertEqual(serial_forward, parallel_forward);
    assertEqual(serial_backward, parallel_backward);
}
