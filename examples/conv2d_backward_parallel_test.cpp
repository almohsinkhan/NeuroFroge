#include "conv2d.h"
#include "execution.h"

#include <cassert>
#include <cmath>
#include <vector>

void assertClose(const Tensor& left, const Tensor& right) {
    assert(left.getShape() == right.getShape());
    for (int i = 0; i < left.size(); i++) {
        assert(std::abs(left.flat(i) - right.flat(i)) < 1e-12);
    }
}

void setWeights(Conv2D& conv) {
    for (int i = 0; i < conv.getWeight().size(); i++) {
        conv.getWeight().flat(i) = 0.1 * (i + 1);
    }
}

int main() {
    Tensor input_base({1, 2, 4, 4});
    for (int i = 0; i < input_base.size(); i++) {
        input_base.flat(i) = 0.2 * (i + 1);
    }
    Tensor input = input_base.broadcastTo({2, 2, 4, 4});

    Tensor dZ_base({1, 3, 3, 3});
    for (int i = 0; i < dZ_base.size(); i++) {
        dZ_base.flat(i) = 0.05 * (i + 1);
    }
    Tensor dZ = dZ_base.broadcastTo({2, 3, 3, 3});

    Conv2D serial_conv(2, 3, 2, 2);
    Conv2D parallel_conv(2, 3, 2, 2);
    setWeights(serial_conv);
    setWeights(parallel_conv);

    execution::setParallelThreshold(1000000);
    Tensor serial_dX = serial_conv.backward(input, dZ);
    Tensor serial_grad_bias = serial_conv.getGradBias();
    Tensor serial_grad_weight = serial_conv.getGradWeight();

    execution::setParallelThreshold(1);
    Tensor parallel_dX = parallel_conv.backward(input, dZ);
    assertClose(serial_dX, parallel_dX);
    assertClose(serial_grad_bias, parallel_conv.getGradBias());
    assertClose(serial_grad_weight, parallel_conv.getGradWeight());

    std::vector<double> first_grad_weight;
    for (int i = 0; i < parallel_conv.getGradWeight().size(); i++) {
        first_grad_weight.push_back(parallel_conv.getGradWeight().flat(i));
    }
    std::vector<double> first_grad_bias;
    for (int i = 0; i < parallel_conv.getGradBias().size(); i++) {
        first_grad_bias.push_back(parallel_conv.getGradBias().flat(i));
    }

    parallel_conv.backward(input, dZ);
    for (int i = 0; i < static_cast<int>(first_grad_bias.size()); i++) {
        assert(
            std::abs(
                parallel_conv.getGradBias().flat(i) -
                2.0 * first_grad_bias[i]
            ) < 1e-12
        );
    }
    for (int i = 0; i < static_cast<int>(first_grad_weight.size()); i++) {
        assert(
            std::abs(
                parallel_conv.getGradWeight().flat(i) -
                2.0 * first_grad_weight[i]
            ) < 1e-12
        );
    }

    parallel_conv.zero_grad();
    parallel_conv.backward(input, dZ);
    for (int i = 0; i < static_cast<int>(first_grad_bias.size()); i++) {
        assert(
            std::abs(
                first_grad_bias[i] -
                parallel_conv.getGradBias().flat(i)
            ) < 1e-12
        );
    }
    for (int i = 0; i < static_cast<int>(first_grad_weight.size()); i++) {
        assert(
            std::abs(
                first_grad_weight[i] -
                parallel_conv.getGradWeight().flat(i)
            ) < 1e-12
        );
    }
}
