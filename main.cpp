#include <iostream>

#include "dataset.h"
#include "linear.h"
#include "activation.h"
#include "softmax.h"
#include "cross_entropy.h"

int argmax(const Tensor& output) {

    int index = 0;
    double max_value = output.flat(0);

    for (int i = 1; i < output.size(); i++) {

        if (output.flat(i) > max_value) {
            max_value = output.flat(i);
            index = i;
        }
    }

    return index;
}

int main() {

    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    int train_size = 800;
    int test_size = dataset.size() - train_size;

    int batch_size = 32;

    Linear layer1(784, 128);
    Linear layer2(128, 10);

    double learning_rate = 0.01;
    int epochs = 50;

    for (int epoch = 0; epoch < epochs; epoch++) {

        double total_loss = 0.0;

        for (int start = 0; start < train_size; start += batch_size) {

            int current_batch_size =
                std::min(batch_size, train_size - start);

            layer1.zero_grad();
            layer2.zero_grad();

            for (int i = start;
                 i < start + current_batch_size;
                 i++) {

                Tensor x = dataset.getImage(i);
                int label = dataset.getLabel(i);

                Tensor y_true({10, 1});

                for (int j = 0; j < 10; j++) {
                    y_true.flat(j) = 0.0;
                }

                y_true.flat(label) = 1.0;

                Tensor z1 = layer1.forward(x);
                Tensor hidden = relu(z1);

                Tensor z2 = layer2.forward(hidden);
                Tensor probabilities = softmax(z2);

                double loss = cross_entropy(
                    y_true,
                    probabilities
                );

                total_loss += loss;

                Tensor dZ2 = cross_entropy_gradient(
                    y_true,
                    probabilities
                );

                Tensor dX2 = layer2.backward(
                    hidden,
                    dZ2
                );

                Tensor relu_grad = relu_derivative(z1);

                Tensor dZ1({128, 1});

                for (int j = 0; j < dX2.size(); j++) {
                    dZ1.flat(j) =
                        dX2.flat(j) * relu_grad.flat(j);
                }

                layer1.backward(
                    x,
                    dZ1
                );
            }

            layer2.update(
                learning_rate,
                current_batch_size
            );

            layer1.update(
                learning_rate,
                current_batch_size
            );
        }

        double average_loss =
            total_loss / train_size;

        std::cout
            << "Epoch "
            << epoch + 1
            << " | Loss: "
            << average_loss
            << "\n";
    }

    int correct = 0;

    for (int i = train_size; i < dataset.size(); i++) {

        Tensor x = dataset.getImage(i);
        int label = dataset.getLabel(i);

        Tensor z1 = layer1.forward(x);
        Tensor hidden = relu(z1);

        Tensor z2 = layer2.forward(hidden);
        Tensor probabilities = softmax(z2);

        int prediction = argmax(probabilities);

        if (prediction == label) {
            correct++;
        }
    }

    double accuracy =
        static_cast<double>(correct) / test_size;

    std::cout
        << "Test Accuracy: "
        << accuracy * 100
        << "%\n";

    return 0;
}