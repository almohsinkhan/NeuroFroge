#pragma once

#include "tensor.h"

class Loss {
    public:
        virtual double forward(
            const Tensor& prediction,
            const Tensor& target
        ) = 0;

        virtual Tensor backward(
            const Tensor& prediction,
            const Tensor& target
        ) = 0;

        virtual ~Loss() = default;
};

