#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <cmath>
#include <vector>

Dataset makeDataset() {
    Dataset dataset("data/images.csv", "data/labels.csv");
    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;

    Tensor image({2, 3, 3});
    for (int index = 0; index < image.size(); index++) {
        image.flat(index) = static_cast<double>(index);
    }
    dataset.images.push_back(image);
    dataset.labels.push_back(4);
    return dataset;
}

int main() {
    Dataset first_dataset = makeDataset();
    Dataset second_dataset = makeDataset();

    AugmentationConfig config;
    config.seed = 123;
    config.horizontal_flip_probability = 1.0;
    config.crop_height = 2;
    config.crop_width = 2;
    config.channel_means = {1.0, 2.0};
    config.channel_stddevs = {2.0, 4.0};
    config.brightness_probability = 1.0;
    config.brightness_min_factor = 1.1;
    config.brightness_max_factor = 1.1;
    config.contrast_probability = 1.0;
    config.contrast_min_factor = 0.9;
    config.contrast_max_factor = 0.9;

    DataLoader configured(
        first_dataset,
        1,
        1,
        false,
        BatchFormat::CNN,
        config
    );
    DataLoader positional(
        second_dataset,
        1,
        1,
        false,
        BatchFormat::CNN,
        config.seed,
        config.horizontal_flip_probability,
        config.crop_height,
        config.crop_width,
        config.channel_means,
        config.channel_stddevs,
        config.brightness_probability,
        config.brightness_min_factor,
        config.brightness_max_factor,
        config.contrast_probability,
        config.contrast_min_factor,
        config.contrast_max_factor
    );

    Batch configured_batch = configured.next();
    Batch positional_batch = positional.next();
    assert(configured_batch.labels == positional_batch.labels);
    assert(configured_batch.images.getShape() ==
           std::vector<int>({1, 2, 2, 2}));
    assert(configured_batch.images.getShape() ==
           positional_batch.images.getShape());
    for (int index = 0; index < configured_batch.images.size(); index++) {
        assert(std::fabs(
            configured_batch.images.flat(index) -
            positional_batch.images.flat(index)
        ) < 1e-12);
    }

    AugmentationConfig defaults;
    assert(defaults.horizontal_flip_probability == 0.0);
    assert(defaults.crop_height == 0);
    assert(defaults.crop_width == 0);
    assert(defaults.brightness_probability == 0.0);
    assert(defaults.contrast_probability == 0.0);
    assert(defaults.channel_means.empty());
    assert(defaults.channel_stddevs.empty());
}
