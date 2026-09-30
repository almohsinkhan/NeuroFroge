#include "tanh.h"

Tensor Tanh::forward(const Tensor& input) {
    return tanh(input);
}

Tensor Tanh::backward(const Tensor& input, const Tensor& grad_output) {
    Tensor grad = tanh_derivative(input);

    for (int i = 0; i < grad.size(); i++) {
        grad.flat(i) *= grad_output.flat(i);
    }

    return grad;
}

