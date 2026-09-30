#include "sigmoid.h"

Tensor Sigmoid::forward(const Tensor& input) {
    return sigmoid(input);
}

Tensor Sigmoid::backward(const Tensor& input, const Tensor& grad_output) {
    Tensor grad = sigmoid_derivative(input);

    for (int i = 0; i < grad.size(); i++) {
        grad.flat(i) *= grad_output.flat(i);
    }

    return grad;
}

