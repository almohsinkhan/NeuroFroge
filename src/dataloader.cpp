#include "dataloader.h"

#include <algorithm>
#include <random>
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
        if (dim <= 0) {
            throw std::invalid_argument(
                "Dataset sample dimensions must be positive"
            );
        }
        sample_size *= dim;
    }

    int channels = 0;
    int height = 0;
    int width = 0;

    if (format == BatchFormat::CNN) {
        if (image_shape.size() == 2) {
            channels = 1;
            height = image_shape[0];
            width = image_shape[1];
        } else if (image_shape.size() == 3) {
            channels = image_shape[0];
            height = image_shape[1];
            width = image_shape[2];
        } else {
            throw std::invalid_argument(
                "DataLoader CNN format requires [height, width] "
                "or [channels, height, width] samples"
            );
        }
    }

    Tensor images = format == BatchFormat::MLP
        ? Tensor({sample_size, current_batch_size})
        : Tensor({
            current_batch_size,
            channels,
            height,
            width
        });

    std::vector<int> labels;


    for (int j = 0; j < current_batch_size; j++) {

        // Position inside shuffled indices
        int dataset_index =
            indices[current_index + j];

        Tensor image =
            dataset.getImage(dataset_index);

        if (image.getShape() != image_shape) {
            throw std::invalid_argument(
                "Dataset samples must have consistent shapes"
            );
        }

        if (format == BatchFormat::MLP) {
            for (int i = 0; i < sample_size; i++) {
                images.flat(
                    i * current_batch_size + j
                ) = image.flat(i);
            }
        } else {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        images({j, c, h, w}) =
                            image.ndim() == 2
                                ? image({h, w})
                                : image({c, h, w});
                    }
                }
            }
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