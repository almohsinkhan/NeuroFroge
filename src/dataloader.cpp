#include "dataloader.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>


DataLoader::DataLoader(
    Dataset& dataset,
    int batch_size,
    int max_samples,
    bool shuffle,
    BatchFormat format
)
    : DataLoader(
          dataset,
          batch_size,
          max_samples,
          shuffle,
          format,
          std::random_device{}()
      )
{
}


DataLoader::DataLoader(
    Dataset& dataset,
    int batch_size,
    int max_samples,
    bool shuffle,
    BatchFormat format,
    std::uint32_t seed,
    double horizontal_flip_probability
)
    : dataset(dataset),
      batch_size(batch_size),
      max_samples(max_samples),
      current_index(0),
      shuffle(shuffle),
      format(format),
      generator(seed),
      horizontal_flip_probability(horizontal_flip_probability)
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

    if (!std::isfinite(horizontal_flip_probability) ||
        horizontal_flip_probability < 0.0 ||
        horizontal_flip_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader horizontal flip probability must be between 0 and 1"
        );
    }

    initializeIndices();

    if (shuffle) {
        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
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

        bool flip = false;
        if (horizontal_flip_probability == 1.0) {
            flip = true;
        } else if (horizontal_flip_probability > 0.0) {
            std::bernoulli_distribution distribution(
                horizontal_flip_probability
            );
            flip = distribution(generator);
        }

        if (format == BatchFormat::MLP) {
            for (int i = 0; i < sample_size; i++) {
                int source_index = i;
                if (flip) {
                    if (image_shape.size() == 2) {
                        int width = image_shape[1];
                        int row = i / width;
                        int column = i % width;
                        source_index =
                            row * width + (width - 1 - column);
                    } else if (image_shape.size() == 3) {
                        int height = image_shape[1];
                        int width = image_shape[2];
                        int channel = i / (height * width);
                        int offset = i % (height * width);
                        int row = offset / width;
                        int column = offset % width;
                        source_index =
                            channel * height * width +
                            row * width +
                            (width - 1 - column);
                    } else {
                        throw std::invalid_argument(
                            "DataLoader horizontal flipping requires "
                            "[height, width] or "
                            "[channels, height, width] samples"
                        );
                    }
                }
                images.flat(
                    i * current_batch_size + j
                ) = image.flat(source_index);
            }
        } else {
            if (flip && image_shape.size() != 2 &&
                image_shape.size() != 3) {
                throw std::invalid_argument(
                    "DataLoader horizontal flipping requires "
                    "[height, width] or "
                    "[channels, height, width] samples"
                );
            }
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int source_width = width - 1 - w;
                        images({j, c, h, w}) =
                            image.ndim() == 2
                                ? image({h, flip ? source_width : w})
                                : image({
                                    c,
                                    h,
                                    flip ? source_width : w
                                });
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
        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );
    }
}