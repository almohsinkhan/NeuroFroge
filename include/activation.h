#pragma once

#include "tensor.h"

Tensor relu(const Tensor& A);
Tensor sigmoid(const Tensor& A);
Tensor tanh(const Tensor& A);

Tensor relu_derivative(const Tensor& A);
Tensor sigmoid_derivative(const Tensor& A);
Tensor tanh_derivative(const Tensor& A);