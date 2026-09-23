#pragma once

#include "tensor.h"

double binary_cross_entropy(
    const Tensor& predictions,
    const Tensor& targets
);

Tensor binary_cross_entropy_backward(
    const Tensor& predictions,
    const Tensor& targets
);