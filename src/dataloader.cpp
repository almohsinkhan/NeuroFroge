#include "dataloader.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

DataLoader::DataLoader(
    Dataset& dataset,
    int batch_size
)
    : DataLoader(
          dataset,
          batch_size,
          static_cast<int>(dataset.size())
      )
{
}


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
    double horizontal_flip_probability,
    int crop_height,
    int crop_width
)
    : dataset(dataset),
      batch_size(batch_size),
      max_samples(max_samples),
      current_index(0),
      shuffle(shuffle),
      format(format),
      generator(seed),
      horizontal_flip_probability(horizontal_flip_probability),
      crop_height(crop_height),
      crop_width(crop_width)
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

    if ((crop_height == 0) != (crop_width == 0) ||
        crop_height < 0 ||
        crop_width < 0) {
        throw std::invalid_argument(
            "DataLoader crop dimensions must both be zero or positive"
        );
    }

    if (crop_height > 0) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader cropping requires [height, width] or "
                "[channels, height, width] samples"
            );
        }
        int source_height = shape[shape.size() - 2];
        int source_width = shape[shape.size() - 1];
        if (crop_height > source_height || crop_width > source_width) {
            throw std::invalid_argument(
                "DataLoader crop dimensions must fit image dimensions"
            );
        }
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
    bool cropping = crop_height > 0;

    if (format == BatchFormat::CNN || cropping ||
        horizontal_flip_probability > 0.0) {
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
                cropping || horizontal_flip_probability > 0.0
                    ? "DataLoader image augmentation requires [height, width] "
                      "or [channels, height, width] samples"
                    : "DataLoader CNN format requires [height, width] "
                      "or [channels, height, width] samples"
            );
        }
    }

    int output_height = cropping ? crop_height : height;
    int output_width = cropping ? crop_width : width;
    int output_sample_size = cropping
        ? channels * output_height * output_width
        : sample_size;

    Tensor images = format == BatchFormat::MLP
        ? Tensor({output_sample_size, current_batch_size})
        : Tensor({
            current_batch_size,
            channels,
            output_height,
            output_width
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

        int crop_top = 0;
        int crop_left = 0;
        if (cropping) {
            std::uniform_int_distribution<int> top_distribution(
                0,
                height - crop_height
            );
            std::uniform_int_distribution<int> left_distribution(
                0,
                width - crop_width
            );
            crop_top = top_distribution(generator);
            crop_left = left_distribution(generator);
        }

        if (format == BatchFormat::MLP) {
            for (int i = 0; i < output_sample_size; i++) {
                int source_index = i;
                if (cropping || flip) {
                    int channel = channels == 1
                        ? 0
                        : i / (output_height * output_width);
                    int offset = i % (output_height * output_width);
                    int row = offset / output_width;
                    int column = offset % output_width;
                    int source_row = cropping ? crop_top + row : row;
                    int source_column = cropping
                        ? crop_left + (flip
                            ? output_width - 1 - column
                            : column)
                        : (flip ? width - 1 - column : column);
                    source_index = image_shape.size() == 2
                        ? source_row * width + source_column
                        : channel * height * width +
                          source_row * width + source_column;
                }
                images.flat(
                    i * current_batch_size + j
                ) = image.flat(source_index);
            }
        } else {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        int source_row = cropping ? crop_top + h : h;
                        int source_width = cropping
                            ? crop_left + (flip
                                ? output_width - 1 - w
                                : w)
                            : (flip ? width - 1 - w : w);
                        images({j, c, h, w}) =
                            image.ndim() == 2
                                ? image({source_row, source_width})
                                : image({
                                    c,
                                    source_row,
                                    source_width
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