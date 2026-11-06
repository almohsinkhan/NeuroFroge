#include "relu.h"
#include "execution.h"

#include <cassert>

Tensor ReLU::forward(const Tensor& input) {
    Tensor output(input.getShape());

    execution::parallelFor(
        static_cast<std::size_t>(input.size()),
        [&](std::size_t index) {
            output.flat(index) =
                input.flat(index) > 0.0
                    ? input.flat(index)
                    : 0.0;
        }
    );

    return output;
}

Tensor ReLU::backward(
    const Tensor& input,
    const Tensor& grad_output
) {
    assert(input.getShape() == grad_output.getShape());
    Tensor grad(input.getShape());

    execution::parallelFor(
        static_cast<std::size_t>(input.size()),
        [&](std::size_t index) {
            grad.flat(index) =
                input.flat(index) > 0.0
                    ? grad_output.flat(index)
                    : 0.0;
        }
    );

    return grad;
}