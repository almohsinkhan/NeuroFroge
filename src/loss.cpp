#include "loss.h"

#include <cassert>
#include <cmath>


double binary_cross_entropy(
    const Tensor& predictions,
    const Tensor& targets
) {

    assert(predictions.getShape() == targets.getShape());

    double loss = 0.0;

    for (int i = 0; i < predictions.size(); i++) {

        double p = predictions.flat(i);
        double y = targets.flat(i);

        assert(p > 0.0 && p < 1.0);

        loss += -(
            y * std::log(p) +
            (1 - y) * std::log(1 - p)
        );
    }

    return loss / predictions.size();
}


Tensor binary_cross_entropy_backward(
    const Tensor& predictions,
    const Tensor& targets
) {

    assert(predictions.getShape() == targets.getShape());

    Tensor grad(predictions.getShape());

    for (int i = 0; i < predictions.size(); i++) {

        grad.flat(i) =
            predictions.flat(i) -
            targets.flat(i);
    }

    return grad;
}