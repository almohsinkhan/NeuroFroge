#include "linear.h"

#include "tensor_ops.h"

#include <cassert>
#include <stdexcept>


Linear::Linear(
    int in_features,
    int out_features,
    const InitMethod init_method
)
    : weight({out_features, in_features}),
      bias({out_features}),
      grad_weight({out_features, in_features}),
      grad_bias({out_features})
{
    if (init_method == InitMethod::XAVIER) {

        weight = xavier_initialization(
            in_features,
            out_features
        );

    } else if (init_method == InitMethod::HE) {

        weight = he_initialization(
            in_features,
            out_features
        );

    } else if (init_method == InitMethod::RANDOM_NORMAL) {

        weight = random_normal_initialization(
            in_features,
            out_features
        );

    } else {

        throw std::invalid_argument(
            "Invalid initialization method"
        );
    }

    bias = zeros(out_features);
}


Tensor Linear::forward(const Tensor& input) {

    assert(input.ndim() == 2);

    assert(
        input.getShape()[0] ==
        weight.getShape()[1]
    );

    Tensor output = matmul(weight, input);

    for (int i = 0; i < output.size(); i++) {
        output({i, 0}) += bias({i});
    }

    return output;
}


Tensor Linear::backward(
    const Tensor& input,
    const Tensor& dZ
) {

    Tensor input_T = transpose(input);

    Tensor current_grad_weight =
        matmul(dZ, input_T);

    for (int i = 0; i < grad_weight.size(); i++) {
        grad_weight.flat(i) +=
            current_grad_weight.flat(i);
    }

    for (int i = 0; i < bias.size(); i++) {
        grad_bias.flat(i) += dZ.flat(i);
    }

    Tensor weight_T = transpose(weight);

    Tensor dX = matmul(weight_T, dZ);

    return dX;
}


void Linear::update(
    double learning_rate,
    int batch_size
) {

    for (int i = 0; i < weight.size(); i++) {

        weight.flat(i) -=
            learning_rate *
            grad_weight.flat(i) /
            batch_size;
    }

    for (int i = 0; i < bias.size(); i++) {

        bias.flat(i) -=
            learning_rate *
            grad_bias.flat(i) /
            batch_size;
    }
}


void Linear::zero_grad() {

    for (int i = 0; i < grad_weight.size(); i++) {
        grad_weight.flat(i) = 0.0;
    }

    for (int i = 0; i < grad_bias.size(); i++) {
        grad_bias.flat(i) = 0.0;
    }
}


Tensor& Linear::getWeight() {
    return weight;
}


Tensor& Linear::getBias() {
    return bias;
}


Tensor& Linear::getGradWeight() {
    return grad_weight;
}


Tensor& Linear::getGradBias() {
    return grad_bias;
}