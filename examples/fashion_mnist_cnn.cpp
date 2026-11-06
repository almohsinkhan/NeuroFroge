#include "conv2d.h"
#include "cross_entropy.h"
#include "dataloader.h"
#include "dataset.h"
#include "flatten.h"
#include "linear.h"
#include "maxpool2d.h"
#include "relu.h"
#include "sequential.h"
#include "softmax.h"
#include "trainer.h"

#include <iostream>

namespace {

int argmax(
    const Tensor& probabilities,
    int sample,
    int batch_size
) {
    int best_class = 0;
    double best_value = probabilities.flat(sample);

    for (int class_index = 1;
         class_index < probabilities.getShape()[0];
         class_index++) {
        double value =
            probabilities.flat(class_index * batch_size + sample);

        if (value > best_value) {
            best_value = value;
            best_class = class_index;
        }
    }

    return best_class;
}

}

int main() {
    Dataset dataset = Dataset::fromIDX(
        "data/FashionMNIST/raw/train-images-idx3-ubyte",
        "data/FashionMNIST/raw/train-labels-idx1-ubyte"
    );

    Sequential model;
    model.add<Conv2D>(1, 8, 3, 3);
    model.add<ReLU>();
    model.add<MaxPool2D>(2, 2, 2);
    model.add<Flatten>();
    model.add<Linear>(1352, 10);

    CrossEntropyLoss loss;
    Trainer trainer(
        model,
        dataset,
        loss,
        BatchFormat::CNN
    );

    const int epochs = 1;
    const int batch_size = 64;
    const int train_size = static_cast<int>(dataset.size());

    trainer.fit(
        epochs,
        batch_size,
        0.01,
        train_size
    );

    DataLoader loader(
        dataset,
        batch_size,
        train_size,
        false,
        BatchFormat::CNN
    );

    double total_loss = 0.0;
    int correct = 0;
    int samples = 0;

    while (loader.hasNext()) {
        Batch batch = loader.next();
        int current_batch_size = batch.images.getShape()[0];

        Tensor probabilities =
            softmax(model.forward(batch.images));

        Tensor targets({
            dataset.numClasses(),
            current_batch_size
        });

        for (int sample = 0;
             sample < current_batch_size;
             sample++) {
            targets.flat(
                batch.labels[sample] * current_batch_size + sample
            ) = 1.0;

            if (argmax(
                    probabilities,
                    sample,
                    current_batch_size
                ) == batch.labels[sample]) {
                correct++;
            }
        }

        total_loss +=
            loss.forward(probabilities, targets) *
            current_batch_size;
        samples += current_batch_size;
    }

    std::cout
        << "Training loss: "
        << total_loss / samples
        << '\n'
        << "Training accuracy: "
        << static_cast<double>(correct) / samples
        << '\n';
}
