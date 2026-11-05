#include "conv2d.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

void Conv2D::update(double, int) {}

void Conv2D::zero_grad() {}

Tensor& Conv2D::getWeight() {
    return weight;
}

Tensor& Conv2D::getBias() {
    return bias;
}

Tensor& Conv2D::getGradWeight() {
    return grad_weight;
}

Tensor& Conv2D::getGradBias() {
    return grad_bias;
}

double loss(
    Conv2D& conv,
    const Tensor& input,
    const Tensor& dZ
) {
    Tensor output = conv.forward(input);
    double result = 0.0;

    for (int i = 0; i < output.size(); i++) {
        result += output.flat(i) * dZ.flat(i);
    }

    return result;
}

int main() {
    Conv2D conv(1, 1, 2, 2, 1, 0);

    Tensor input({1, 1, 3, 3});
    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = 0.2 * (i + 1);
    }

    conv.getWeight().flat(0) = 0.3;
    conv.getWeight().flat(1) = -0.4;
    conv.getWeight().flat(2) = 0.5;
    conv.getWeight().flat(3) = -0.6;
    conv.getBias().flat(0) = 0.7;

    Tensor dZ({1, 1, 2, 2});
    dZ.flat(0) = 0.8;
    dZ.flat(1) = -0.3;
    dZ.flat(2) = 0.6;
    dZ.flat(3) = 1.1;

    conv.getGradWeight() = Tensor({1, 1, 2, 2});
    conv.getGradBias() = Tensor({1});
    Tensor analytical_dX = conv.backward(input, dZ);

    const double epsilon = 1e-6;
    double max_error = 0.0;

    for (int i = 0; i < conv.getWeight().size(); i++) {
        double original = conv.getWeight().flat(i);

        conv.getWeight().flat(i) = original + epsilon;
        double plus = loss(conv, input, dZ);

        conv.getWeight().flat(i) = original - epsilon;
        double minus = loss(conv, input, dZ);

        conv.getWeight().flat(i) = original;
        double numerical = (plus - minus) / (2.0 * epsilon);
        max_error = std::max(
            max_error,
            std::abs(numerical - conv.getGradWeight().flat(i))
        );
    }

    double original_bias = conv.getBias().flat(0);
    conv.getBias().flat(0) = original_bias + epsilon;
    double plus_bias = loss(conv, input, dZ);
    conv.getBias().flat(0) = original_bias - epsilon;
    double minus_bias = loss(conv, input, dZ);
    conv.getBias().flat(0) = original_bias;

    double numerical_bias =
        (plus_bias - minus_bias) / (2.0 * epsilon);
    max_error = std::max(
        max_error,
        std::abs(numerical_bias - conv.getGradBias().flat(0))
    );

    for (int i = 0; i < input.size(); i++) {
        double original = input.flat(i);

        input.flat(i) = original + epsilon;
        double plus = loss(conv, input, dZ);

        input.flat(i) = original - epsilon;
        double minus = loss(conv, input, dZ);

        input.flat(i) = original;
        double numerical = (plus - minus) / (2.0 * epsilon);
        max_error = std::max(
            max_error,
            std::abs(numerical - analytical_dX.flat(i))
        );
    }

    std::cout << "Maximum gradient error: " << max_error << '\n';
    assert(max_error < 1e-8);
}
