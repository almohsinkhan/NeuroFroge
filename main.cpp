#include <iostream>
#include <algorithm>
#include <cstdlib>

#include "dataset.h"
#include "linear.h"
#include "activation.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "sequential.h"
#include "trainer.h"
#include "relu.h"

int argmax(const Tensor &output)
{
    int index = 0;
    double max_value = output.flat(0);

    for (int i = 1; i < output.size(); i++)
    {
        if (output.flat(i) > max_value)
        {
            max_value = output.flat(i);
            index = i;
        }
    }

    return index;
}

Dataset loadDataset(
    const std::string& image_path,
    const std::string& label_path
)
{
    try {
        return Dataset(image_path, label_path);
    } catch (const std::exception& error) {
        std::cerr
            << "Failed to load dataset: "
            << error.what()
            << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc != 1 && argc != 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " [images.csv labels.csv]\n";
        return 2;
    }

    const std::string image_path =
        argc == 3 ? argv[1] : "data/images.csv";
    const std::string label_path =
        argc == 3 ? argv[2] : "data/labels.csv";

    Dataset dataset = loadDataset(image_path, label_path);

    // number of classes in the dataset
    int num_classes = dataset.numClasses();

    Sequential model;

    const int feature_count = dataset.getImage(0).size();
    model.add<Linear>(feature_count, 128);
    model.add<ReLU>();
    model.add<Linear>(128, num_classes);

    model.print_summary();

    CrossEntropyLoss loss;
    Trainer trainer(model, dataset, loss);

    // Test Sequential forward pass
    Tensor image = dataset.getImage(0);
    Tensor x({feature_count, 1});

    for (int i = 0; i < image.size(); i++) {
        x.flat(i) = image.flat(i);
    }
    Tensor output = model.forward(x);

    std::cout << "Output shape: ";
    output.printShape();

    double learning_rate = 0.01;
    int epochs = 5;
    int batch_size = 32;
    int train_size =
        std::min(800, static_cast<int>(dataset.size()) - 1);
    int test_size = dataset.size() - train_size;

    trainer.fit(epochs, batch_size, learning_rate, train_size);

    int correct = 0;

    for (int i = train_size; i < dataset.size(); i++)
    {

        int label = dataset.getLabel(i);
        
        Tensor image = dataset.getImage(i);
        Tensor x({feature_count, 1});

        for (int j = 0; j < image.size(); j++) {
            x.flat(j) = image.flat(j);
        }

        Tensor logits = model.forward(x);
        Tensor probabilities = softmax(logits);

        int prediction = argmax(probabilities);

        if (prediction == label)
        {
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