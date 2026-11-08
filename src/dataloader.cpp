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
    int crop_width,
    const std::vector<double>& channel_means,
    const std::vector<double>& channel_stddevs,
    double brightness_probability,
    double brightness_min_factor,
    double brightness_max_factor,
    double contrast_probability,
    double contrast_min_factor,
    double contrast_max_factor
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
      crop_width(crop_width),
      channel_means(channel_means),
      channel_stddevs(channel_stddevs),
      brightness_probability(brightness_probability),
      brightness_min_factor(brightness_min_factor),
      brightness_max_factor(brightness_max_factor),
      contrast_probability(contrast_probability),
      contrast_min_factor(contrast_min_factor),
      contrast_max_factor(contrast_max_factor)
{
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "DataLoader batch_size must be positive"
        );
    }

    if (!std::isfinite(contrast_probability) ||
        contrast_probability < 0.0 ||
        contrast_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader contrast probability must be between 0 and 1"
        );
    }

    if (!std::isfinite(contrast_min_factor) ||
        !std::isfinite(contrast_max_factor) ||
        contrast_min_factor <= 0.0 ||
        contrast_max_factor <= 0.0 ||
        contrast_min_factor > contrast_max_factor) {
        throw std::invalid_argument(
            "DataLoader contrast factors must be positive and ordered"
        );
    }

    if (max_samples <= 0 ||
        max_samples > static_cast<int>(dataset.size())) {
        throw std::invalid_argument(
            "DataLoader max_samples must be within dataset size"
        );
    }

    if (contrast_probability > 0.0) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader contrast adjustment requires [height, width] "
                "or [channels, height, width] samples"
            );
        }
    }

    if (!std::isfinite(horizontal_flip_probability) ||
        horizontal_flip_probability < 0.0 ||
        horizontal_flip_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader horizontal flip probability must be between 0 and 1"
        );
    }

    if (!std::isfinite(brightness_probability) ||
        brightness_probability < 0.0 ||
        brightness_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader brightness probability must be between 0 and 1"
        );
    }

    if (!std::isfinite(brightness_min_factor) ||
        !std::isfinite(brightness_max_factor) ||
        brightness_min_factor <= 0.0 ||
        brightness_max_factor <= 0.0 ||
        brightness_min_factor > brightness_max_factor) {
        throw std::invalid_argument(
            "DataLoader brightness factors must be positive and ordered"
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

    if (channel_means.empty() != channel_stddevs.empty()) {
        throw std::invalid_argument(
            "DataLoader channel means and standard deviations must both "
            "be provided"
        );
    }

    if (!channel_means.empty()) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader channel normalization requires [height, width] "
                "or [channels, height, width] samples"
            );
        }
        int expected_channels = shape.size() == 2 ? 1 : shape[0];
        if (channel_means.size() !=
                static_cast<std::size_t>(expected_channels) ||
            channel_stddevs.size() !=
                static_cast<std::size_t>(expected_channels)) {
            throw std::invalid_argument(
                "DataLoader channel normalization parameters must match "
                "the sample channel count"
            );
        }
        for (std::size_t channel = 0;
             channel < channel_means.size();
             channel++) {
            if (!std::isfinite(channel_means[channel]) ||
                !std::isfinite(channel_stddevs[channel]) ||
                channel_stddevs[channel] <= 0.0) {
                throw std::invalid_argument(
                    "DataLoader channel means must be finite and standard "
                    "deviations must be positive and finite"
                );
            }
        }
    }

    if (brightness_probability > 0.0) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader brightness adjustment requires [height, width] "
                "or [channels, height, width] samples"
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
        horizontal_flip_probability > 0.0 ||
        contrast_probability > 0.0) {
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
                cropping || horizontal_flip_probability > 0.0 ||
                contrast_probability > 0.0
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

    auto normalize = [this](double value, int channel) {
        if (channel_means.empty()) {
            return value;
        }
        return (value - channel_means[channel]) /
               channel_stddevs[channel];
    };

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

        double brightness_factor = 1.0;
        if (brightness_probability == 1.0) {
            brightness_factor = std::uniform_real_distribution<double>(
                brightness_min_factor,
                brightness_max_factor
            )(generator);
        } else if (brightness_probability > 0.0) {
            std::bernoulli_distribution apply_brightness(
                brightness_probability
            );
            if (apply_brightness(generator)) {
                brightness_factor = std::uniform_real_distribution<double>(
                    brightness_min_factor,
                    brightness_max_factor
                )(generator);
            }
        }

        double contrast_factor = 1.0;
        if (contrast_probability == 1.0) {
            contrast_factor = std::uniform_real_distribution<double>(
                contrast_min_factor,
                contrast_max_factor
            )(generator);
        } else if (contrast_probability > 0.0) {
            std::bernoulli_distribution apply_contrast(
                contrast_probability
            );
            if (apply_contrast(generator)) {
                contrast_factor = std::uniform_real_distribution<double>(
                    contrast_min_factor,
                    contrast_max_factor
                )(generator);
            }
        }

        std::vector<double> contrast_means(channels, 0.0);
        if (contrast_factor != 1.0) {
            for (int c = 0; c < channels; c++) {
                int channel_size = height * width;
                int channel_offset = image_shape.size() == 2
                    ? 0
                    : c * channel_size;
                for (int index = 0; index < channel_size; index++) {
                    contrast_means[c] +=
                        image.flat(channel_offset + index);
                }
                contrast_means[c] /= channel_size;
            }
        }

        auto transform = [&](double value, int channel) {
            value *= brightness_factor;
            if (contrast_factor != 1.0) {
                double mean = contrast_means[channel] * brightness_factor;
                value = mean +
                        contrast_factor *
                        (value - mean);
            }
            return value;
        };

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
                int channel = 0;
                if (!channel_means.empty() &&
                    image_shape.size() == 3) {
                    channel = i / (image_shape[1] * image_shape[2]);
                }
                images.flat(i * current_batch_size + j) =
                    normalize(
                        transform(image.flat(source_index), channel),
                        channel
                    );
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
                        double value =
                            image.ndim() == 2
                                ? image({source_row, source_width})
                                : image({
                                    c,
                                    source_row,
                                    source_width
                                });
                        images({j, c, h, w}) =
                            normalize(transform(value, c), c);
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