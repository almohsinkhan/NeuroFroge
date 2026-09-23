#include <iostream>
#include "dataset.h"
#include "linear.h"

int main() {

    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    std::cout << "Dataset size: "
              << dataset.size() << "\n";

    // Get one image
    Tensor x = dataset.getImage(0);
    int label = dataset.getLabel(0);

    std::cout << "Image shape: ";
    x.printShape();

    std::cout << "True label: "
              << label << "\n";


    // Neural network
    Linear layer1(784, 128);
    Linear layer2(128, 10);

    // Forward pass
    Tensor hidden = layer1.forward(x);
    Tensor output = layer2.forward(hidden);

    std::cout << "Output shape: ";
    output.printShape();

    return 0;
}