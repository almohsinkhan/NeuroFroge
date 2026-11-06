#include "dataloader.h"

#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>


DataLoader::DataLoader(
    Dataset& dataset,
    int batch_size,
    int max_samples,
    bool shuffle,
    BatchFormat format
)
    : dataset(dataset),
      batch_size(batch_size),
      max_samples(max_samples),
      current_index(0),
      shuffle(shuffle),
      format(format)
{
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "DataLoader batch_size must be positive"
        );
    }

    if (max_samples <= 0 ||
        max_samples > static_cast<int>(dataset.size())) {
        throw std::invalid_argument(
            "DataLoader max_samples must be within dataset size"
        );
    }

    initializeIndices();

    // Shuffle the first epoch
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(
            indices.begin(),
            indices.end(),
            g
        );
    }
}


void DataLoader::initializeIndices() {

    indices.clear();

    for (int i = 0; i < max_samples; i++) {
        indices.push_back(i);
    }
}


bool DataLoader::hasNext() const {

    return current_index < max_samples;
}


Batch DataLoader::next() {

    int remaining =
        max_samples - current_index;

    int current_batch_size =
        std::min(batch_size, remaining);


    std::vector<int> image_shape =
        dataset.sampleShape();

    int sample_size = 1;

    for (int dim : image_shape) {
        sample_size *= dim;
    }

    if (sample_size != 784) {
        throw std::invalid_argument(
            "DataLoader expects 784-pixel samples"
        );
    }

    Tensor images = format == BatchFormat::MLP
        ? Tensor({sample_size, current_batch_size})
        : Tensor({current_batch_size, 1, 28, 28});

    std::vector<int> labels;


    for (int j = 0; j < current_batch_size; j++) {

        // Position inside shuffled indices
        int dataset_index =
            indices[current_index + j];

        Tensor image =
            dataset.getImage(dataset_index);


        if (format == BatchFormat::MLP) {
            for (int i = 0; i < sample_size; i++) {
                images.flat(
                    i * current_batch_size + j
                ) = image.flat(i);
            }
        } else if (image.ndim() == 2) {
            for (int h = 0; h < 28; h++) {
                for (int w = 0; w < 28; w++) {
                    images({j, 0, h, w}) =
                        image({h, w});
                }
            }
        } else if (image.ndim() == 3 &&
                   image.getShape()[0] == 1 &&
                   image.getShape()[1] == 28 &&
                   image.getShape()[2] == 28) {
            for (int h = 0; h < 28; h++) {
                for (int w = 0; w < 28; w++) {
                    images({j, 0, h, w}) =
                        image({0, h, w});
                }
            }
        } else {
            throw std::invalid_argument(
                "DataLoader CNN format requires 28x28 or 1x28x28 samples"
            );
        }


        labels.push_back(
            dataset.getLabel(dataset_index)
        );
    }


    current_index += current_batch_size;

    return {images, labels};
}


void DataLoader::reset() {

    current_index = 0;

    if (shuffle) {

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(
            indices.begin(),
            indices.end(),
            g
        );
    }
}