#include "cross_entropy.h"
#include <cmath>

double cross_entropy(
    const Tensor& y_true,
    const Tensor& y_pred
) {
    int num_classes = y_pred.getShape()[0];
    int batch_size = y_pred.getShape()[1];

    double total_loss = 0.0;

    for (int j = 0; j < batch_size; j++) {

        for (int i = 0; i < num_classes; i++) {

            double target =
                y_true.flat(i * batch_size + j);

            double prediction =
                y_pred.flat(i * batch_size + j);

            total_loss -=
                target * std::log(prediction);
        }
    }

    return total_loss / batch_size;
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

double CrossEntropyLoss::forward(
    const Tensor& prediction,
    const Tensor& target
) {
    return cross_entropy(target, prediction);
}

Tensor CrossEntropyLoss::backward(
    const Tensor& prediction,
    const Tensor& target
) {
    return cross_entropy_gradient(target, prediction);
}

