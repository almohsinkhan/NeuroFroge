#pragma once

#include "module.h"
#include "tensor.h"

#include <vector>


class Flatten : public Module {
private:
    std::vector<int> original_input_shape;

public:
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
        return "Flatten";
    }
};
