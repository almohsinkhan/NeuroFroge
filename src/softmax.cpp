#include "softmax.h"
#include <cmath>

Tensor softmax(const Tensor& input) {

    Tensor output(input.getShape());

    double max_value = input.flat(0);

    // Find maximum value
    for (int i = 1; i < input.size(); i++) {
        if (input.flat(i) > max_value) {
            max_value = input.flat(i);
        }
    }

    // Calculate exponentials
    double sum = 0.0;

    for (int i = 0; i < input.size(); i++) {
        output.flat(i) = std::exp(input.flat(i) - max_value);
        sum += output.flat(i);
    }

    // Normalize
    for (int i = 0; i < input.size(); i++) {
        output.flat(i) /= sum;
    }

    return output;
}