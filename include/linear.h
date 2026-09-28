#pragma once

#include "module.h"
#include "tensor.h"
#include "initialization.h"


class Linear : public Module {
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

    Tensor forward(const Tensor& input) override;

    Tensor backward(
        const Tensor& input,
        const Tensor& dZ
    ) override;

    void update(
        double learning_rate,
        int batch_size
    ) override;

    void zero_grad() override;

    Tensor& getWeight();
    Tensor& getBias();

    Tensor& getGradWeight();
    Tensor& getGradBias();
};