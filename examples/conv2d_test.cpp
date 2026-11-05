#define private public
#include "conv2d.h"
#undef private

#include <cassert>
#include <cmath>

Tensor Conv2D::backward(const Tensor&, const Tensor&) {
    return Tensor({1});
}

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

int main() {
    Conv2D conv(1, 1, 2, 2, 1, 0);

    Tensor input({1, 1, 3, 3});
    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = i + 1;
    }

    conv.weight({0, 0, 0, 0}) = 1.0;
    conv.weight({0, 0, 0, 1}) = 0.0;
    conv.weight({0, 0, 1, 0}) = 0.0;
    conv.weight({0, 0, 1, 1}) = 1.0;
    conv.bias({0}) = 0.0;

    Tensor output = conv.forward(input);

    assert(output.getShape() == std::vector<int>({1, 1, 2, 2}));

    const double expected[] = {6.0, 8.0, 12.0, 14.0};
    for (int i = 0; i < output.size(); i++) {
        assert(std::abs(output.flat(i) - expected[i]) < 1e-12);
    }
}
