#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

Dataset makeDataset(const std::vector<int>& shape, int count = 1) {
    Dataset dataset("data/images.csv", "data/labels.csv");
    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;
    for (int sample = 0; sample < count; sample++) {
        Tensor image(shape);
        for (int index = 0; index < image.size(); index++) {
            image.flat(index) = sample * 100.0 + index + 1.0;
        }
        dataset.images.push_back(image);
        dataset.labels.push_back(sample);
    }
    return dataset;
}

void assertNear(double actual, double expected) {
    assert(std::fabs(actual - expected) < 1e-12);
}

int main() {
    Dataset grayscale = makeDataset({3, 3});

    AugmentationConfig resize;
    resize.seed = 9;
    resize.resize_height = 2;
    resize.resize_width = 2;
    DataLoader resize_loader(
        grayscale, 1, 1, false, BatchFormat::CNN, resize
    );
    Batch resized = resize_loader.next();
    assert(resized.images.getShape() ==
           std::vector<int>({1, 1, 2, 2}));
    assertNear(resized.images({0, 0, 0, 0}), 1.0);
    assertNear(resized.images({0, 0, 0, 1}), 3.0);
    assertNear(resized.images({0, 0, 1, 0}), 7.0);
    assertNear(resized.images({0, 0, 1, 1}), 9.0);

    AugmentationConfig translated;
    translated.seed = 10;
    translated.translation_min_horizontal = 1;
    translated.translation_max_horizontal = 1;
    translated.translation_min_vertical = 1;
    translated.translation_max_vertical = 1;
    DataLoader translated_loader(
        grayscale, 1, 1, false, BatchFormat::CNN, translated
    );
    Batch shifted = translated_loader.next();
    assertNear(shifted.images({0, 0, 0, 0}), 0.0);
    assertNear(shifted.images({0, 0, 1, 1}), 1.0);
    assertNear(shifted.images({0, 0, 2, 2}), 5.0);

    AugmentationConfig vertical;
    vertical.seed = 11;
    vertical.vertical_flip_probability = 1.0;
    DataLoader vertical_loader(
        grayscale, 1, 1, false, BatchFormat::CNN, vertical
    );
    Batch flipped = vertical_loader.next();
    assertNear(flipped.images({0, 0, 0, 0}), 7.0);
    assertNear(flipped.images({0, 0, 2, 2}), 3.0);

    Dataset rgb = makeDataset({2, 3, 3});
    AugmentationConfig blur;
    blur.seed = 12;
    blur.blur_kernel_size = 3;
    blur.blur_sigma = 1.0;
    DataLoader blur_loader(
        rgb, 1, 1, false, BatchFormat::CNN, blur
    );
    Batch blurred = blur_loader.next();
    assert(blurred.images.getShape() ==
           std::vector<int>({1, 2, 3, 3}));
    assert(blurred.images({0, 0, 1, 1}) > 1.0);
    assert(blurred.images({0, 1, 1, 1}) > 10.0);

    Dataset noise_first_dataset = makeDataset({3, 3}, 2);
    Dataset noise_second_dataset = makeDataset({3, 3}, 2);
    AugmentationConfig noise;
    noise.seed = 13;
    noise.noise_mean = 0.25;
    noise.noise_stddev = 0.5;
    DataLoader noise_first(
        noise_first_dataset, 1, 2, true, BatchFormat::CNN, noise
    );
    DataLoader noise_second(
        noise_second_dataset, 1, 2, true, BatchFormat::CNN, noise
    );
    while (noise_first.hasNext()) {
        Batch first = noise_first.next();
        Batch second = noise_second.next();
        assert(first.labels == second.labels);
        for (int index = 0; index < first.images.size(); index++) {
            assertNear(first.images.flat(index), second.images.flat(index));
        }
    }

    AugmentationConfig composed = noise;
    composed.vertical_flip_probability = 1.0;
    composed.horizontal_flip_probability = 1.0;
    composed.rotation_probability = 1.0;
    composed.rotation_min_degrees = 0.0;
    composed.rotation_max_degrees = 0.0;
    composed.translation_min_horizontal = -1;
    composed.translation_max_horizontal = 1;
    composed.translation_min_vertical = -1;
    composed.translation_max_vertical = 1;
    composed.resize_height = 4;
    composed.resize_width = 4;
    composed.blur_kernel_size = 3;
    composed.blur_sigma = 0.8;
    composed.brightness_probability = 1.0;
    composed.brightness_min_factor = 1.0;
    composed.brightness_max_factor = 1.0;
    DataLoader composed_loader(
        rgb, 1, 1, false, BatchFormat::CNN, composed
    );
    Batch composed_batch = composed_loader.next();
    assert(composed_batch.images.getShape() ==
           std::vector<int>({1, 2, 4, 4}));
    assert(composed_batch.labels == std::vector<int>({0}));

    for (const auto& invalid : std::vector<AugmentationConfig>{
        [] {
            AugmentationConfig config;
            config.vertical_flip_probability = -0.1;
            return config;
        }(),
        [] {
            AugmentationConfig config;
            config.translation_min_horizontal = 2;
            config.translation_max_horizontal = 1;
            return config;
        }(),
        [] {
            AugmentationConfig config;
            config.resize_height = 3;
            return config;
        }(),
        [] {
            AugmentationConfig config;
            config.noise_stddev = -1.0;
            return config;
        }(),
        [] {
            AugmentationConfig config;
            config.blur_kernel_size = 2;
            config.blur_sigma = 1.0;
            return config;
        }(),
        [] {
            AugmentationConfig config;
            config.blur_kernel_size = 3;
            config.blur_sigma = 0.0;
            return config;
        }()
    }) {
        bool rejected = false;
        try {
            DataLoader loader(
                grayscale, 1, 1, false, BatchFormat::CNN, invalid
            );
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
}
