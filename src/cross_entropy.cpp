#include "cross_entropy.h"
#include <cmath>

double cross_entropy(
    const Tensor& y_true,
    const Tensor& y_pred
) {
    double loss = 0.0;

    for (int i = 0; i < y_true.size(); i++) {
        loss -= y_true.flat(i) *
                std::log(y_pred.flat(i));
    }

    return loss;
}

Tensor cross_entropy_gradient(
    const Tensor& y_true,
    const Tensor& y_pred
) {
    Tensor grad(y_pred.getShape());

    for (int i = 0; i < y_pred.size(); i++) {
        grad.flat(i) =
            y_pred.flat(i) - y_true.flat(i);
    }

    return grad;
}