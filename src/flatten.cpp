#include "flatten.h"
#include "execution.h"

#include <cassert>
#include <stdexcept>

Tensor Flatten::forward(const Tensor& input) {
    assert(input.ndim() == 4);

    original_input_shape = input.getShape();

    int batch_size = original_input_shape[0];
    int features =
        original_input_shape[1] *
        original_input_shape[2] *
        original_input_shape[3];

    Tensor output({features, batch_size});

    execution::parallelFor(
        static_cast<std::size_t>(input.size()),
        [&](std::size_t index) {
            int n = index / features;
            int feature = index % features;

            output.flat(feature * batch_size + n) =
                input.flat(index);
        }
    );

    return output;
}

Tensor Flatten::backward(
    const Tensor&,
    const Tensor& dZ
) {
    if (original_input_shape.empty()) {
        throw std::logic_error(
            "Flatten::backward() called before forward()"
        );
    }

    int batch_size = original_input_shape[0];
    int features =
        original_input_shape[1] *
        original_input_shape[2] *
        original_input_shape[3];

    assert(dZ.ndim() == 2);
    assert(dZ.getShape()[0] == features);
    assert(dZ.getShape()[1] == batch_size);

    Tensor dX(original_input_shape);

    execution::parallelFor(
        static_cast<std::size_t>(dZ.size()),
        [&](std::size_t index) {
            int feature = index / batch_size;
            int n = index % batch_size;

            dX.flat(n * features + feature) =
                dZ.flat(index);
        }
    );

    return dX;
}

void Flatten::update(
    double,
    int
) {}

void Flatten::zero_grad() {}
