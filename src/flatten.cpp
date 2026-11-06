#include "flatten.h"

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

    for (int n = 0; n < batch_size; n++) {
        for (int feature = 0; feature < features; feature++) {
            output.flat(feature * batch_size + n) =
                input.flat(n * features + feature);
        }
    }

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

    for (int n = 0; n < batch_size; n++) {
        for (int feature = 0; feature < features; feature++) {
            dX.flat(n * features + feature) =
                dZ.flat(feature * batch_size + n);
        }
    }

    return dX;
}

void Flatten::update(
    double,
    int
) {}

void Flatten::zero_grad() {}
