#include <iostream>
#include <algorithm>

#include "dataset.h"
#include "linear.h"
#include "activation.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "sequential.h"
#include "trainer.h"

// Adapter to use your existing ReLU function as a Module
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return relu(input);
    }

    Tensor backward(
        const Tensor& input,
        const Tensor& grad_output
    ) override {
        Tensor grad = relu_derivative(input);

        for (int i = 0; i < grad.size(); i++) {
            grad.flat(i) *= grad_output.flat(i);
        }

        return grad;
    }

    void zero_grad() override {}

    void update(double learning_rate, int batch_size) override {}
};

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


    Linear layer1(784, 128);
    ReLU relu_layer;
    Linear layer2(128, 10);

    Sequential model;

    model.add(&layer1);
    model.add(&relu_layer);
    model.add(&layer2);

    CrossEntropyLoss loss;
    Trainer trainer(model, dataset, loss);  
      
    // Test Sequential forward pass
    Tensor x = dataset.getImage(0);

    Tensor output = model.forward(x);

    std::cout << "Output shape: ";
    output.printShape();

    double learning_rate = 0.01;
    int epochs = 50;
    int batch_size = 32;
    int train_size = 800;
    int test_size = dataset.size() - train_size;


    trainer.fit(epochs, batch_size, learning_rate, train_size);


    int correct = 0;

    for (int i = train_size; i < dataset.size(); i++) {

        Tensor x = dataset.getImage(i);
        int label = dataset.getLabel(i);

        Tensor logits = model.forward(x);
        Tensor probabilities = softmax(logits);

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