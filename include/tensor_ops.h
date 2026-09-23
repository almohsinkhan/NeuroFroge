#pragma once

#include "tensor.h"

Tensor matmul(const Tensor& A, const Tensor& B);

Tensor add(const Tensor& A, const Tensor& B);

Tensor multiply(const Tensor& A, const Tensor& B);

Tensor subtract(const Tensor& A, const Tensor& B);

Tensor scale(const Tensor& A, double scalar);

Tensor sum(const Tensor& A);

Tensor mean(const Tensor& A);

Tensor transpose(const Tensor& A);