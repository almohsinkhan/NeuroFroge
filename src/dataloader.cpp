#include "dataloader.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

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
    const AugmentationConfig& augmentation
)
    : DataLoader(
          dataset,
          batch_size,
          max_samples,
          shuffle,
          format,
          augmentation.seed,
          augmentation.horizontal_flip_probability,
          augmentation.crop_height,
          augmentation.crop_width,
          augmentation.channel_means,
          augmentation.channel_stddevs,
          augmentation.brightness_probability,
          augmentation.brightness_min_factor,
          augmentation.brightness_max_factor,
          augmentation.contrast_probability,
          augmentation.contrast_min_factor,
          augmentation.contrast_max_factor,
          augmentation.rotation_probability,
          augmentation.rotation_min_degrees,
          augmentation.rotation_max_degrees,
          augmentation.vertical_flip_probability,
          augmentation.translation_min_horizontal,
          augmentation.translation_max_horizontal,
          augmentation.translation_min_vertical,
          augmentation.translation_max_vertical,
          augmentation.resize_height,
          augmentation.resize_width,
          augmentation.noise_mean,
          augmentation.noise_stddev,
          augmentation.blur_kernel_size,
          augmentation.blur_sigma
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
    double contrast_max_factor,
    double rotation_probability,
    double rotation_min_degrees,
    double rotation_max_degrees,
    double vertical_flip_probability,
    int translation_min_horizontal,
    int translation_max_horizontal,
    int translation_min_vertical,
    int translation_max_vertical,
    int resize_height,
    int resize_width,
    double noise_mean,
    double noise_stddev,
    int blur_kernel_size,
    double blur_sigma
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
      contrast_max_factor(contrast_max_factor),
      rotation_probability(rotation_probability),
      rotation_min_degrees(rotation_min_degrees),
      rotation_max_degrees(rotation_max_degrees),
      vertical_flip_probability(vertical_flip_probability),
      translation_min_horizontal(translation_min_horizontal),
      translation_max_horizontal(translation_max_horizontal),
      translation_min_vertical(translation_min_vertical),
      translation_max_vertical(translation_max_vertical),
      resize_height(resize_height),
      resize_width(resize_width),
      noise_mean(noise_mean),
      noise_stddev(noise_stddev),
      blur_kernel_size(blur_kernel_size),
      blur_sigma(blur_sigma)
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

    if (!std::isfinite(rotation_probability) ||
        rotation_probability < 0.0 ||
        rotation_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader rotation probability must be between 0 and 1"
        );
    }

    if (!std::isfinite(rotation_min_degrees) ||
        !std::isfinite(rotation_max_degrees) ||
        rotation_min_degrees > rotation_max_degrees) {
        throw std::invalid_argument(
            "DataLoader rotation angles must be finite and ordered"
        );
    }

    if (!std::isfinite(vertical_flip_probability) ||
        vertical_flip_probability < 0.0 ||
        vertical_flip_probability > 1.0) {
        throw std::invalid_argument(
            "DataLoader vertical flip probability must be between 0 and 1"
        );
    }

    if (translation_min_horizontal > translation_max_horizontal ||
        translation_min_vertical > translation_max_vertical) {
        throw std::invalid_argument(
            "DataLoader translation ranges must be ordered"
        );
    }

    if ((resize_height == 0) != (resize_width == 0) ||
        resize_height < 0 ||
        resize_width < 0) {
        throw std::invalid_argument(
            "DataLoader resize dimensions must both be zero or positive"
        );
    }

    if (!std::isfinite(noise_mean) ||
        !std::isfinite(noise_stddev) ||
        noise_stddev < 0.0) {
        throw std::invalid_argument(
            "DataLoader noise parameters must be finite and non-negative"
        );
    }

    if (blur_kernel_size != 0 &&
        (blur_kernel_size < 1 || blur_kernel_size % 2 == 0)) {
        throw std::invalid_argument(
            "DataLoader blur kernel size must be zero or a positive odd value"
        );
    }
    if (!std::isfinite(blur_sigma) ||
        (blur_kernel_size > 0 && blur_sigma <= 0.0) ||
        (blur_kernel_size == 0 && blur_sigma < 0.0)) {
        throw std::invalid_argument(
            "DataLoader blur sigma must be positive when blur is enabled"
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

    if (brightness_probability > 0.0 ||
        vertical_flip_probability > 0.0 ||
        translation_min_horizontal != 0 ||
        translation_max_horizontal != 0 ||
        translation_min_vertical != 0 ||
        translation_max_vertical != 0 ||
        resize_height > 0 ||
        blur_kernel_size > 0 ||
        noise_mean != 0.0 ||
        noise_stddev > 0.0) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader image augmentation requires [height, width] "
                "or [channels, height, width] samples"
            );
        }
    }

    if (rotation_probability > 0.0) {
        std::vector<int> shape = dataset.sampleShape();
        if (shape.size() != 2 && shape.size() != 3) {
            throw std::invalid_argument(
                "DataLoader rotation requires [height, width] or "
                "[channels, height, width] samples"
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
        contrast_probability > 0.0 ||
        rotation_probability > 0.0 ||
        vertical_flip_probability > 0.0 ||
        translation_min_horizontal != 0 ||
        translation_max_horizontal != 0 ||
        translation_min_vertical != 0 ||
        translation_max_vertical != 0 ||
        resize_height > 0 ||
        noise_mean != 0.0 ||
        noise_stddev > 0.0 ||
        blur_kernel_size > 0) {
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
                contrast_probability > 0.0 ||
                rotation_probability > 0.0
                    ? "DataLoader image augmentation requires [height, width] "
                      "or [channels, height, width] samples"
                    : "DataLoader CNN format requires [height, width] "
                      "or [channels, height, width] samples"
            );
        }
    }

    int base_height = cropping ? crop_height : height;
    int base_width = cropping ? crop_width : width;
    int output_height = resize_height > 0 ? resize_height : base_height;
    int output_width = resize_width > 0 ? resize_width : base_width;
    int output_sample_size = channels > 0
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

        double rotation_degrees = 0.0;
        if (rotation_probability == 1.0) {
            rotation_degrees = std::uniform_real_distribution<double>(
                rotation_min_degrees,
                rotation_max_degrees
            )(generator);
        } else if (rotation_probability > 0.0) {
            std::bernoulli_distribution apply_rotation(
                rotation_probability
            );
            if (apply_rotation(generator)) {
                rotation_degrees = std::uniform_real_distribution<double>(
                    rotation_min_degrees,
                    rotation_max_degrees
                )(generator);
            }
        }

        bool vertical_flip = false;
        if (vertical_flip_probability == 1.0) {
            vertical_flip = true;
        } else if (vertical_flip_probability > 0.0) {
            vertical_flip = std::bernoulli_distribution(
                vertical_flip_probability
            )(generator);
        }

        int translation_horizontal = 0;
        int translation_vertical = 0;
        if (translation_min_horizontal != translation_max_horizontal) {
            translation_horizontal = std::uniform_int_distribution<int>(
                translation_min_horizontal,
                translation_max_horizontal
            )(generator);
        } else {
            translation_horizontal = translation_min_horizontal;
        }
        if (translation_min_vertical != translation_max_vertical) {
            translation_vertical = std::uniform_int_distribution<int>(
                translation_min_vertical,
                translation_max_vertical
            )(generator);
        } else {
            translation_vertical = translation_min_vertical;
        }

        double rotation_radians =
            rotation_degrees * std::acos(-1.0) / 180.0;
        double rotation_cosine = std::cos(rotation_radians);
        double rotation_sine = std::sin(rotation_radians);
        double spatial_center_row = (base_height - 1) / 2.0;
        double spatial_center_column = (base_width - 1) / 2.0;

        auto sourceCoordinates = [&](int row, int column) {
            int scaled_row = resize_height > 0
                ? static_cast<int>(
                    (row + 0.5) * base_height / output_height
                )
                : row;
            int scaled_column = resize_width > 0
                ? static_cast<int>(
                    (column + 0.5) * base_width / output_width
                )
                : column;
            int translated_row = scaled_row - translation_vertical;
            int translated_column =
                scaled_column - translation_horizontal;
            double centered_row = translated_row - spatial_center_row;
            double centered_column =
                translated_column - spatial_center_column;
            int rotated_row = static_cast<int>(std::lround(
                spatial_center_row +
                rotation_cosine * centered_row +
                rotation_sine * centered_column
            ));
            int rotated_column = static_cast<int>(std::lround(
                spatial_center_column +
                -rotation_sine * centered_row +
                rotation_cosine * centered_column
            ));
            if (vertical_flip) {
                rotated_row = base_height - 1 - rotated_row;
            }
            if (flip) {
                rotated_column = base_width - 1 - rotated_column;
            }
            return std::pair<int, int>(rotated_row, rotated_column);
        };

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

        std::vector<double> spatial_values(
            channels * output_height * output_width,
            0.0
        );
        if (channels == 0) {
            for (int i = 0; i < sample_size; i++) {
                images.flat(i * current_batch_size + j) =
                    normalize(transform(image.flat(i), 0), 0);
            }
            labels.push_back(dataset.getLabel(dataset_index));
            continue;
        }
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    auto source = sourceCoordinates(h, w);
                    double value = 0.0;
                    if (source.first >= 0 &&
                        source.first < base_height &&
                        source.second >= 0 &&
                        source.second < base_width) {
                        int source_row = cropping
                            ? crop_top + source.first
                            : source.first;
                        int source_column = cropping
                            ? crop_left + source.second
                            : source.second;
                        value = image.ndim() == 2
                            ? image({source_row, source_column})
                            : image({c, source_row, source_column});
                    }
                    spatial_values[
                        (c * output_height + h) * output_width + w
                    ] = value;
                }
            }
        }

        if (blur_kernel_size > 0) {
            std::vector<double> kernel(blur_kernel_size);
            int radius = blur_kernel_size / 2;
            double kernel_sum = 0.0;
            for (int i = 0; i < blur_kernel_size; i++) {
                double distance = i - radius;
                kernel[i] = std::exp(
                    -(distance * distance) /
                    (2.0 * blur_sigma * blur_sigma)
                );
                kernel_sum += kernel[i];
            }
            for (double& value : kernel) {
                value /= kernel_sum;
            }
            std::vector<double> horizontal(spatial_values.size(), 0.0);
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        for (int k = 0; k < blur_kernel_size; k++) {
                            int source_w = w + k - radius;
                            if (source_w >= 0 && source_w < output_width) {
                                horizontal[
                                    (c * output_height + h) *
                                    output_width + w
                                ] += kernel[k] * spatial_values[
                                    (c * output_height + h) *
                                    output_width + source_w
                                ];
                            }
                        }
                    }
                }
            }
            std::vector<double> blurred(spatial_values.size(), 0.0);
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        for (int k = 0; k < blur_kernel_size; k++) {
                            int source_h = h + k - radius;
                            if (source_h >= 0 && source_h < output_height) {
                                blurred[
                                    (c * output_height + h) *
                                    output_width + w
                                ] += kernel[k] * horizontal[
                                    (c * output_height + source_h) *
                                    output_width + w
                                ];
                            }
                        }
                    }
                }
            }
            spatial_values.swap(blurred);
        }

        if (noise_stddev > 0.0 || noise_mean != 0.0) {
            std::normal_distribution<double> noise(noise_mean, noise_stddev);
            for (double& value : spatial_values) {
                value += noise(generator);
            }
        }

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    double value = spatial_values[
                        (c * output_height + h) * output_width + w
                    ];
                    value = normalize(transform(value, c), c);
                    if (format == BatchFormat::MLP) {
                        int index = (c * output_height + h) *
                            output_width + w;
                        images.flat(index * current_batch_size + j) = value;
                    } else {
                        images({j, c, h, w}) = value;
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