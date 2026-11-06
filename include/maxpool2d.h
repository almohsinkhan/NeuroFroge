#pragma once

#include "module.h"
#include "tensor.h"

#include <vector>


class MaxPool2D : public Module {
private:
    int kernel_height;
    int kernel_width;
    int stride;

    std::vector<int> max_positions;

public:
    MaxPool2D(
        int kernel_height,
        int kernel_width,
        int stride
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

    std::string name() const override {
        return "MaxPool2D";
    }
};
