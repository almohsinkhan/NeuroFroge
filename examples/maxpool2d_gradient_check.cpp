#include "maxpool2d.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

double loss(
    MaxPool2D& pool,
    const Tensor& input,
    const Tensor& dZ
) {
    Tensor output = pool.forward(input);
    double result = 0.0;

    for (int i = 0; i < output.size(); i++) {
        result += output.flat(i) * dZ.flat(i);
    }

    return result;
}

int main() {
    MaxPool2D pool(2, 2, 2);

    Tensor input({1, 1, 4, 4});
    const double values[] = {
        0.1, 1.3, 0.2, 1.4,
        2.5, 0.9, 2.6, 0.8,
        0.3, 3.7, 0.4, 3.8,
        4.5, 0.6, 4.6, 0.7
    };

    for (int i = 0; i < input.size(); i++) {
        input.flat(i) = values[i];
    }

    Tensor dZ({1, 1, 2, 2});
    dZ.flat(0) = 0.8;
    dZ.flat(1) = -0.3;
    dZ.flat(2) = 0.6;
    dZ.flat(3) = 1.1;

    pool.forward(input);
    Tensor analytical_dX = pool.backward(input, dZ);
    const double epsilon = 1e-6;
    double max_error = 0.0;

    for (int i = 0; i < input.size(); i++) {
        double original = input.flat(i);

        input.flat(i) = original + epsilon;
        double plus = loss(pool, input, dZ);

        input.flat(i) = original - epsilon;
        double minus = loss(pool, input, dZ);

        input.flat(i) = original;
        double numerical = (plus - minus) / (2.0 * epsilon);

        max_error = std::max(
            max_error,
            std::abs(numerical - analytical_dX.flat(i))
        );
    }

    std::cout << "Maximum gradient error: " << max_error << '\n';
    assert(max_error < 1e-8);
}
