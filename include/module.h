#pragma once

#include "tensor.h"

class Module {
    public:
        /*virtual allows different classes to provide their own implementations of the same function*/

        virtual Tensor forward(const Tensor& input) = 0;

        virtual Tensor backward(
            const Tensor& input,
            const Tensor& dZ
        ) = 0;

        virtual void zero_grad() = 0;

        virtual void update(
            double learning_rate,
            int batch_size
        ) = 0;

        /*
        Later, our Sequential model will store layers through Module* pointers. Since a Linear object is also a Module, 
        we need to ensure it gets destroyed correctly when deleted through its base-class pointer
        */
        virtual ~Module() = default;
};


