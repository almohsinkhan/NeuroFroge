#include <iostream>
#include "trainer.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "dataloader.h"

Trainer::Trainer(
    Sequential& model,
    Dataset& dataset,
    Loss& loss,
    BatchFormat format
)
    : model(model),
      dataset(dataset),
      loss(loss),
      format(format)
{
}

void Trainer::fit(
    int epochs,
    int batch_size,
    double learning_rate,
    int train_size
) {
    int num_classes = dataset.numClasses();

    DataLoader loader(
        dataset,
        batch_size,
        train_size,
        true,
        format
    );

    for (int epoch = 0; epoch < epochs; epoch++) {

        double total_loss = 0.0;
        int total_correct = 0;
        int batch_number = 0;

        loader.reset();

        while (loader.hasNext()) {

            Batch batch = loader.next();
            batch_number++;


            int current_batch_size =
                format == BatchFormat::MLP
                    ? batch.images.getShape()[1]
                    : batch.images.getShape()[0];

            model.zero_grad();

            // Create one-hot targets
            Tensor y_true({
                num_classes,
                current_batch_size
            });

            for (int j = 0; j < current_batch_size; j++) {

                int label = batch.labels[j];

                for (int i = 0; i < num_classes; i++) {

                    y_true.flat(
                        i * current_batch_size + j
                    ) = 0.0;
                }

                y_true.flat(
                    label * current_batch_size + j
                ) = 1.0;
            }

            // Forward entire batch
            Tensor logits =
                model.forward(batch.images);

            // Softmax entire batch
            Tensor probabilities =
                softmax(logits);

        
            // Calculate batch loss
            double loss_value =
                loss.forward(
                    probabilities,
                    y_true
                );

            int batch_correct = 0;
            for (int j = 0; j < current_batch_size; j++) {
                int predicted_class = 0;
                double best_probability = probabilities.flat(j);

                for (int i = 1; i < num_classes; i++) {
                    double probability =
                        probabilities.flat(i * current_batch_size + j);

                    if (probability > best_probability) {
                        best_probability = probability;
                        predicted_class = i;
                    }
                }

                if (predicted_class == batch.labels[j]) {
                    batch_correct++;
                }
            }

            // Calculate gradient
            Tensor dZ =
                loss.backward(
                    probabilities,
                    y_true
                );

            // Backward entire batch
            model.backward(dZ);

            // Accumulate loss
            total_loss +=
                loss_value * current_batch_size;
            total_correct += batch_correct;

            // Update once for this batch
            model.update(
                learning_rate,
                current_batch_size
            );

            if (batch_number % 5 == 0) {
                std::cout
                    << "Epoch "
                    << epoch + 1
                    << "/"
                    << epochs
                    << " | Batch "
                    << batch_number
                    << " | Loss: "
                    << loss_value
                    << " | Accuracy: "
                    << static_cast<double>(batch_correct) /
                           current_batch_size
                    << std::endl;
            }
        }

        double average_loss =
            total_loss / train_size;
        double average_accuracy =
            static_cast<double>(total_correct) / train_size;

        std::cout
            << "Epoch "
            << epoch + 1
            << "/"
            << epochs
            << " | Loss: "
            << average_loss
            << " | Accuracy: "
            << average_accuracy
            << std::endl;
    }
}