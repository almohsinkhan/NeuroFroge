#pragma once 

#include "activation.h"

class Tanh : public Activation {
    public:
        Tensor forward(const Tensor& input) override;

        Tensor backward(
            const Tensor& input,
            const Tensor& grad_output
        ) override;

        std::string name() const override {
            return "Tanh";
        }
};
