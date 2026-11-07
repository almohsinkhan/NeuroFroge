#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

Dataset makeDataset(
    const std::vector<int>& shape,
    int count
) {
    Dataset dataset("data/images.csv", "data/labels.csv");
    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;

    for (int sample = 0; sample < count; sample++) {
        Tensor image(shape);
        for (int index = 0; index < image.size(); index++) {
            image.flat(index) = sample * 1000 + index;
        }
        dataset.images.push_back(image);
        dataset.labels.push_back(sample);
    }
    return dataset;
}

int main() {
    Dataset grayscale = makeDataset({4, 5}, 2);
    const std::uint32_t seed = 41;
    std::mt19937 expected_generator(seed);
    std::uniform_int_distribution<int> top_distribution(0, 2);
    std::uniform_int_distribution<int> left_distribution(0, 2);
    int top = top_distribution(expected_generator);
    int left = left_distribution(expected_generator);
    int second_top = top_distribution(expected_generator);
    int second_left = left_distribution(expected_generator);

    DataLoader crop_loader(
        grayscale,
        2,
        2,
        false,
        BatchFormat::CNN,
        seed,
        0.0,
        2,
        3
    );
    Batch cropped = crop_loader.next();
    assert(cropped.images.getShape() == std::vector<int>({
        2, 1, 2, 3
    }));
    assert(cropped.labels == std::vector<int>({0, 1}));
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 3; column++) {
            assert(
                cropped.images({0, 0, row, column}) ==
                (top + row) * 5 + left + column
            );
            assert(
                cropped.images({1, 0, row, column}) ==
                1000 + (second_top + row) * 5 + second_left + column
            );
        }
    }

    Dataset rgb = makeDataset({3, 4, 5}, 1);
    DataLoader rgb_loader(
        rgb,
        1,
        1,
        false,
        BatchFormat::CNN,
        seed,
        0.0,
        2,
        3
    );
    Batch rgb_batch = rgb_loader.next();
    assert(rgb_batch.images.getShape() == std::vector<int>({
        1, 3, 2, 3
    }));
    for (int channel = 0; channel < 3; channel++) {
        assert(
            rgb_batch.images({0, channel, 0, 0}) ==
            channel * 20 + top * 5 + left
        );
    }

    Dataset combined = makeDataset({4, 5}, 1);
    std::mt19937 combined_generator(seed);
    int combined_top = std::uniform_int_distribution<int>(
        0, 2
    )(combined_generator);
    int combined_left = std::uniform_int_distribution<int>(
        0, 2
    )(combined_generator);
    DataLoader combined_loader(
        combined,
        1,
        1,
        false,
        BatchFormat::CNN,
        seed,
        1.0,
        2,
        3
    );
    Batch combined_batch = combined_loader.next();
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 3; column++) {
            assert(
                combined_batch.images({0, 0, row, column}) ==
                (combined_top + row) * 5 +
                combined_left + (2 - column)
            );
        }
    }

    Dataset first_dataset = makeDataset({4, 5}, 5);
    Dataset second_dataset = makeDataset({4, 5}, 5);
    DataLoader first(
        first_dataset, 2, 5, true, BatchFormat::CNN, 99, 0.5, 3, 4
    );
    DataLoader second(
        second_dataset, 2, 5, true, BatchFormat::CNN, 99, 0.5, 3, 4
    );
    while (first.hasNext() && second.hasNext()) {
        Batch first_batch = first.next();
        Batch second_batch = second.next();
        assert(first_batch.labels == second_batch.labels);
        assert(first_batch.images.getShape() == second_batch.images.getShape());
        for (int index = 0; index < first_batch.images.size(); index++) {
            assert(first_batch.images.flat(index) ==
                   second_batch.images.flat(index));
        }
    }

    DataLoader disabled(
        grayscale, 2, 2, false, BatchFormat::MLP, 7, 0.0
    );
    Batch unchanged = disabled.next();
    assert(unchanged.images.getShape() == std::vector<int>({20, 2}));
    assert(unchanged.images.flat(0) == 0.0);

    for (const auto& dimensions : std::vector<std::pair<int, int>>{
        {0, 2}, {-1, 2}, {5, 5}
    }) {
        bool rejected = false;
        try {
            DataLoader invalid(
                grayscale,
                1,
                1,
                false,
                BatchFormat::CNN,
                1,
                0.0,
                dimensions.first,
                dimensions.second
            );
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
}
