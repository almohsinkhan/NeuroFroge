#include "relu.h"

Tensor ReLU::forward(const Tensor& input) {
    return relu(input);
}

Tensor ReLU::backward(
    const Tensor& input,
    const Tensor& grad_output
) {
    Tensor grad = relu_derivative(input);

    for (int i = 0; i < grad.size(); i++) {
        grad.flat(i) *= grad_output.flat(i);
    }

    return grad;
}