#include <iostream>

#include "tensor.h"
#include "tensor_ops.h"
#include "activation.h"
#include "linear.h"
#include "loss.h"

int main() {

    // Create linear layers
    Linear layer1(2, 3, InitMethod::XAVIER);
    Linear layer2(3, 1, InitMethod::XAVIER);

    // XOR inputs and targets
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4] = {
        0, 1, 1, 0
    };

    // Training parameters
    double learning_rate = 0.1;
    int epochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {

        double total_loss = 0.0;

        // Reset gradients
        layer1.zero_grad();
        layer2.zero_grad();

        // Loop over each sample
        for (int sample = 0; sample < 4; sample++) {

            Tensor input({2, 1});

            input({0, 0}) = inputs[sample][0];
            input({1, 0}) = inputs[sample][1];

            Tensor target({1, 1});
            target({0, 0}) = targets[sample];

            // Forward pass
            Tensor z1 = layer1.forward(input);
            Tensor hidden = relu(z1);

            Tensor z2 = layer2.forward(hidden);
            Tensor prediction = sigmoid(z2);

            // Loss
            double loss =
                binary_cross_entropy(prediction, target);

            total_loss += loss;

            // Backward pass
            Tensor dZ2 =
                binary_cross_entropy_backward(
                    prediction,
                    target
                );

            Tensor dHidden =
                layer2.backward(hidden, dZ2);

            Tensor relu_grad =
                relu_derivative(z1);

            Tensor dZ1 =
                multiply(dHidden, relu_grad);

            Tensor dInput =
                layer1.backward(input, dZ1);
        }

        // Update parameters
        layer1.update(learning_rate, 4);
        layer2.update(learning_rate, 4);

        if (epoch % 1000 == 0) {
            std::cout
                << "Epoch " << epoch
                << " Loss: "
                << total_loss / 4
                << "\n";
        }
    }

    // Predictions
    std::cout << "\nXOR Predictions\n";

    for (int sample = 0; sample < 4; sample++) {

        Tensor input({2, 1});

        input({0, 0}) = inputs[sample][0];
        input({1, 0}) = inputs[sample][1];

        Tensor z1 = layer1.forward(input);
        Tensor hidden = relu(z1);

        Tensor z2 = layer2.forward(hidden);
        Tensor prediction = sigmoid(z2);

        std::cout
            << inputs[sample][0]
            << " XOR "
            << inputs[sample][1]
            << " = "
            << prediction.flat(0)
            << "\n";
    }

    return 0;
}