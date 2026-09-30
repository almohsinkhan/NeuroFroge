#include <iostream>
#include "trainer.h"
#include "softmax.h"
#include "cross_entropy.h"

Trainer::Trainer(Sequential& model, Dataset& dataset, Loss& loss)
    : model(model), dataset(dataset), loss(loss) {}

void Trainer::fit(int epochs, int batch_size, double learning_rate, int train_size) {
    int num_classes = dataset.numClasses();

    for (int epoch = 0; epoch < epochs; epoch++) {

        double total_loss = 0.0;

        for (int start = 0; start < train_size; start += batch_size){
            int current_batch_size = std::min(batch_size, train_size - start);

            model.zero_grad();

            for(int i = start; i < start + current_batch_size; i++){
                Tensor x = dataset.getImage(i);
                int label = dataset.getLabel(i);

                Tensor y_true({num_classes, 1});

                for (int j = 0; j < num_classes; j++) {
                    y_true.flat(j) = 0.0;
                }

                y_true.flat(label) = 1.0;

                Tensor z2 = model.forward(x);
                Tensor probabilities = softmax(z2);

                double loss_value = loss.forward(probabilities, y_true);

                Tensor dZ2 = loss.backward(probabilities, y_true);

                model.backward(dZ2);

                total_loss += loss_value;
            }

            model.update(learning_rate, current_batch_size);
        }

        double average_loss = total_loss / train_size;

        std::cout << "Epoch "
                << epoch + 1
                << "/" << epochs
                << " | Loss: "
                << average_loss
                << std::endl;
    }
}