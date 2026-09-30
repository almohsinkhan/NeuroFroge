#pragma once

#include "tensor.h"
#include "module.h"

// Activation functions
Tensor relu(const Tensor& A);
Tensor sigmoid(const Tensor& A);
Tensor tanh(const Tensor& A);

Tensor relu_derivative(const Tensor& A);
Tensor sigmoid_derivative(const Tensor& A);
Tensor tanh_derivative(const Tensor& A);

// Base activation module
class Activation : public Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;

    virtual Tensor backward(
        const Tensor& input,
        const Tensor& grad_output
    ) = 0;

    void zero_grad() override {}

    void update(
        double learning_rate,
        int batch_size
    ) override {}
};