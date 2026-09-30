#pragma once

#include "tensor.h"
#include "loss.h"

double cross_entropy(
    const Tensor& y_true,
    const Tensor& y_pred
);

Tensor cross_entropy_gradient(
    const Tensor& y_true,
    const Tensor& y_pred
);


class CrossEntropyLoss : public Loss {
public:
    double forward(
        const Tensor& prediction,
        const Tensor& target
    ) override ;


    Tensor backward(
        const Tensor& prediction,
        const Tensor& target
    ) override ;
};  