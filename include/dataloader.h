#pragma once

#include "dataset.h"
#include "tensor.h"
#include <cstdint>
#include <random>
#include <vector>

struct Batch {
    Tensor images;
    std::vector<int> labels;
};

enum class BatchFormat {
    MLP,
    CNN
};

struct AugmentationConfig {
    std::uint32_t seed = std::random_device{}();
    double horizontal_flip_probability = 0.0;
    int crop_height = 0;
    int crop_width = 0;
    std::vector<double> channel_means;
    std::vector<double> channel_stddevs;
    double brightness_probability = 0.0;
    double brightness_min_factor = 1.0;
    double brightness_max_factor = 1.0;
    double contrast_probability = 0.0;
    double contrast_min_factor = 1.0;
    double contrast_max_factor = 1.0;
    double rotation_probability = 0.0;
    double rotation_min_degrees = 0.0;
    double rotation_max_degrees = 0.0;
    double vertical_flip_probability = 0.0;
    int translation_min_horizontal = 0;
    int translation_max_horizontal = 0;
    int translation_min_vertical = 0;
    int translation_max_vertical = 0;
    int resize_height = 0;
    int resize_width = 0;
    double noise_mean = 0.0;
    double noise_stddev = 0.0;
    int blur_kernel_size = 0;
    double blur_sigma = 0.0;
};

class DataLoader {

private:
    Dataset& dataset;

    int batch_size;
    int max_samples;
    int current_index;

    bool shuffle;
    BatchFormat format;
    double horizontal_flip_probability;
    int crop_height;
    int crop_width;
    std::vector<double> channel_means;
    std::vector<double> channel_stddevs;
    double brightness_probability;
    double brightness_min_factor;
    double brightness_max_factor;
    double contrast_probability;
    double contrast_min_factor;
    double contrast_max_factor;
    double rotation_probability;
    double rotation_min_degrees;
    double rotation_max_degrees;
    double vertical_flip_probability;
    int translation_min_horizontal;
    int translation_max_horizontal;
    int translation_min_vertical;
    int translation_max_vertical;
    int resize_height;
    int resize_width;
    double noise_mean;
    double noise_stddev;
    int blur_kernel_size;
    double blur_sigma;

    std::vector<int> indices;
    std::mt19937 generator;

    void initializeIndices();

public:
    DataLoader(
        Dataset& dataset,
        int batch_size
    );

    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle = true,
        BatchFormat format = BatchFormat::MLP
    );

    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle,
        BatchFormat format,
        const AugmentationConfig& augmentation
    );

    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle,
        BatchFormat format,
        std::uint32_t seed,
        double horizontal_flip_probability = 0.0,
        int crop_height = 0,
        int crop_width = 0,
        const std::vector<double>& channel_means = {},
        const std::vector<double>& channel_stddevs = {},
        double brightness_probability = 0.0,
        double brightness_min_factor = 1.0,
        double brightness_max_factor = 1.0,
        double contrast_probability = 0.0,
        double contrast_min_factor = 1.0,
        double contrast_max_factor = 1.0,
        double rotation_probability = 0.0,
        double rotation_min_degrees = 0.0,
        double rotation_max_degrees = 0.0,
        double vertical_flip_probability = 0.0,
        int translation_min_horizontal = 0,
        int translation_max_horizontal = 0,
        int translation_min_vertical = 0,
        int translation_max_vertical = 0,
        int resize_height = 0,
        int resize_width = 0,
        double noise_mean = 0.0,
        double noise_stddev = 0.0,
        int blur_kernel_size = 0,
        double blur_sigma = 0.0
    );

    bool hasNext() const;

    Batch next();

    void reset();
};