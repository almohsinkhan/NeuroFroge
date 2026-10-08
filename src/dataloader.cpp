#include "dataloader.h"

#include <algorithm>
#include <random>
#include <iostream>


DataLoader::DataLoader(
    Dataset& dataset,
    int batch_size,
    int max_samples,
    bool shuffle
)
    : dataset(dataset),
      batch_size(batch_size),
      max_samples(max_samples),
      current_index(0),
      shuffle(shuffle)
{
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


    Tensor images({
        sample_size,
        current_batch_size
    });

    std::vector<int> labels;


    for (int j = 0; j < current_batch_size; j++) {

        // Position inside shuffled indices
        int dataset_index =
            indices[current_index + j];

        Tensor image =
            dataset.getImage(dataset_index);


        for (int i = 0; i < sample_size; i++) {

            images.flat(
                i * current_batch_size + j
            ) = image.flat(i);
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