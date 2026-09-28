#include <iostream>

#include "dataset.h"
#include "linear.h"
#include "activation.h"
#include "softmax.h"
#include "cross_entropy.h"

int main() {

    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    std::cout << "Dataset size: "
              << dataset.size() << "\n";

    Tensor x = dataset.getImage(0);
    int label = dataset.getLabel(0);

    std::cout << "Image shape: ";
    x.printShape();

    std::cout << "True label: "
              << label << "\n";

    Linear layer1(784, 128);
    Linear layer2(128, 10);

    Tensor z1 = layer1.forward(x);
    Tensor hidden = relu(z1);
    Tensor output = layer2.forward(hidden);

    std::cout << "Output shape: ";
    output.printShape();

    Tensor probabilities = softmax(output);

    std::cout << "Predicted probabilities: ";

    for (int i = 0; i < probabilities.size(); i++) {
        std::cout << probabilities.flat(i) << " ";
    }

    std::cout << "\n";

    Tensor y_true({10, 1});

    for (int i = 0; i < 10; i++) {
        y_true.flat(i) = 0.0;
    }

    y_true.flat(label) = 1.0;

    double loss = cross_entropy(
        y_true,
        probabilities
    );

    std::cout << "Cross-entropy loss: "
              << loss << "\n";

    Tensor dZ2 = cross_entropy_gradient(
        y_true,
        probabilities
    );

    std::cout << "dZ2: ";

    for (int i = 0; i < dZ2.size(); i++) {
        std::cout << dZ2.flat(i) << " ";
    }

    std::cout << "\n";

    Tensor dX2 = layer2.backward(
        hidden,
        dZ2
    );

    std::cout << "dX2 shape: ";
    dX2.printShape();

    Tensor relu_grad = relu_derivative(z1);

    Tensor dZ1({128, 1});

    for (int i = 0; i < dX2.size(); i++) {
        dZ1.flat(i) =
            dX2.flat(i) * relu_grad.flat(i);
    }

    Tensor dX1 = layer1.backward(
        x,
        dZ1
    );

    
    std::cout << "dX1 shape: ";
    dX1.printShape();

    return 0;
}