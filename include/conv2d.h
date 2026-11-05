#pragma once

#include "module.h"
#include "tensor.h"
#include "initialization.h"


class Conv2D : public Module {
private:
    Tensor weight;
    Tensor bias;

    Tensor grad_weight;
    Tensor grad_bias;

    int in_channels;
    int out_channels;
    int kernel_height;
    int kernel_width;
    int stride;
    int padding;

public:
    Conv2D(
        int in_channels,
        int out_channels,
        int kernel_height,
        int kernel_width,
        int stride = 1,
        int padding = 0,
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

    std::string name() const override {
        return "Conv2D";
    }
};
