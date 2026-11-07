#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <cmath>
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
            image.flat(index) = sample * 100.0 + index;
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
    Dataset grayscale = makeDataset({2, 2}, 1);
    DataLoader gray_loader(
        grayscale,
        1,
        1,
        false,
        BatchFormat::CNN,
        7,
        0.0,
        0,
        0,
        {1.0},
        {2.0}
    );
    Batch gray_batch = gray_loader.next();
    assert(gray_batch.labels == std::vector<int>({0}));
    assertNear(gray_batch.images({0, 0, 0, 0}), -0.5);
    assertNear(gray_batch.images({0, 0, 1, 1}), 1.0);

    Dataset rgb = makeDataset({3, 1, 2}, 1);
    DataLoader rgb_loader(
        rgb,
        1,
        1,
        false,
        BatchFormat::CNN,
        7,
        0.0,
        0,
        0,
        {0.0, 1.0, 2.0},
        {1.0, 2.0, 4.0}
    );
    Batch rgb_batch = rgb_loader.next();
    assert(rgb_batch.labels == std::vector<int>({0}));
    assertNear(rgb_batch.images({0, 0, 0, 0}), 0.0);
    assertNear(rgb_batch.images({0, 1, 0, 0}), 0.5);
    assertNear(rgb_batch.images({0, 2, 0, 0}), 0.5);
    assertNear(rgb_batch.images({0, 2, 0, 1}), 0.75);

    DataLoader composed(
        grayscale,
        1,
        1,
        false,
        BatchFormat::CNN,
        19,
        1.0,
        1,
        1,
        {1.0},
        {2.0}
    );
    Batch composed_batch = composed.next();
    assert(composed_batch.images.getShape() == std::vector<int>({
        1, 1, 1, 1
    }));
    assertNear(composed_batch.images({0, 0, 0, 0}), -0.5);

    Dataset first_dataset = makeDataset({2, 2, 2}, 4);
    Dataset second_dataset = makeDataset({2, 2, 2}, 4);
    DataLoader first(
        first_dataset,
        2,
        4,
        true,
        BatchFormat::CNN,
        123,
        0.5,
        1,
        1,
        {1.0, 2.0},
        {2.0, 3.0}
    );
    DataLoader second(
        second_dataset,
        2,
        4,
        true,
        BatchFormat::CNN,
        123,
        0.5,
        1,
        1,
        {1.0, 2.0},
        {2.0, 3.0}
    );
    while (first.hasNext() && second.hasNext()) {
        Batch first_batch = first.next();
        Batch second_batch = second.next();
        assert(first_batch.labels == second_batch.labels);
        for (int index = 0; index < first_batch.images.size(); index++) {
            assertNear(
                first_batch.images.flat(index),
                second_batch.images.flat(index)
            );
        }
    }

    for (const auto& parameters :
         std::vector<std::pair<std::vector<double>, std::vector<double>>>{
             {{}, {1.0}},
             {{0.0}, {}},
             {{0.0, 1.0}, {1.0, 1.0}},
             {{0.0}, {0.0}},
             {{0.0}, {-1.0}}
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
                0,
                0,
                parameters.first,
                parameters.second
            );
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
}
