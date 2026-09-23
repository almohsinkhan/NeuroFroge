#pragma once

#include "tensor.h"

double cross_entropy(const Tensor& y_true, const Tensor& y_pred);

Tensor cross_entropy_gradient(const Tensor& y_true, const Tensor& y_pred);