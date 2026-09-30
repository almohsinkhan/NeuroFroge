#pragma once

#include "activation.h"

class ReLU : public Activation {
    public:
        Tensor forward(const Tensor& input) override;

        Tensor backward(
            const Tensor& input,
            const Tensor& grad_output
        ) override;

};