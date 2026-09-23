#pragma once

#include "tensor.h"
#include "initialization.h"

class Linear {
private:
    Tensor weight;
    Tensor bias;

    Tensor grad_weight;
    Tensor grad_bias;

public:
    Linear(
        int in_features,
        int out_features,
        const InitMethod init_method = InitMethod::RANDOM_NORMAL
    );

    Tensor forward(const Tensor& input);

    Tensor backward(
        const Tensor& input,
        const Tensor& dZ
    );

    void update(
        double learning_rate,
        int batch_size
    );

    void zero_grad();

    Tensor& getWeight();
    Tensor& getBias();

    Tensor& getGradWeight();
    Tensor& getGradBias();
};