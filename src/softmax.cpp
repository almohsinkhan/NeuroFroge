#include "softmax.h"
#include <cmath>

Tensor softmax(const Tensor& input) {

    Tensor output(input.getShape());

    int num_classes = input.getShape()[0];
    int batch_size = input.getShape()[1];

    for (int j = 0; j < batch_size; j++) {

        // Find maximum for this sample
        double max_value = input.flat(j);

        for (int i = 1; i < num_classes; i++) {

            double value =
                input.flat(i * batch_size + j);

            if (value > max_value) {
                max_value = value;
            }
        }

        // Calculate exponentials
        double sum = 0.0;

        for (int i = 0; i < num_classes; i++) {

            double value =
                std::exp(
                    input.flat(i * batch_size + j)
                    - max_value
                );

            output.flat(i * batch_size + j) = value;

            sum += value;
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {

            output.flat(i * batch_size + j)
                /= sum;
        }
    }

    return output;
}