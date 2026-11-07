#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <cmath>
#include <vector>

Dataset makeImageDataset(
    const std::vector<int>& shape,
    int sample_count
) {
    Dataset dataset("data/images.csv", "data/labels.csv");
    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;

    for (int sample = 0; sample < sample_count; sample++) {
        Tensor image(shape);
        for (int index = 0; index < image.size(); index++) {
            image.flat(index) = sample * 1000 + index;
        }
        dataset.images.push_back(image);
        dataset.labels.push_back(sample);
    }

    return dataset;
}

void assertClose(double actual, double expected) {
    assert(std::fabs(actual - expected) < 1e-12);
}

int main() {
    Dataset grayscale = makeImageDataset({2, 3}, 2);

    DataLoader flipped_mlp(
        grayscale,
        2,
        2,
        false,
        BatchFormat::MLP,
        7,
        1.0
    );
    Batch mlp_batch = flipped_mlp.next();
    assert(mlp_batch.labels == std::vector<int>({0, 1}));
    assertClose(mlp_batch.images({0, 0}), 2.0);
    assertClose(mlp_batch.images({1, 0}), 1.0);
    assertClose(mlp_batch.images({2, 0}), 0.0);
    assertClose(mlp_batch.images({3, 0}), 5.0);
    assertClose(mlp_batch.images({4, 0}), 4.0);
    assertClose(mlp_batch.images({5, 0}), 3.0);
    assertClose(mlp_batch.images({0, 1}), 1002.0);
    assertClose(mlp_batch.images({5, 1}), 1003.0);

    DataLoader unchanged(
        grayscale,
        2,
        2,
        false,
        BatchFormat::CNN,
        7,
        0.0
    );
    Batch unchanged_batch = unchanged.next();
    assertClose(unchanged_batch.images({0, 0, 0, 0}), 0.0);
    assertClose(unchanged_batch.images({0, 0, 0, 2}), 2.0);
    assertClose(unchanged_batch.images({0, 0, 1, 0}), 3.0);
    assertClose(unchanged_batch.images({1, 0, 1, 2}), 1005.0);

    Dataset rgb = makeImageDataset({3, 2, 3}, 2);
    DataLoader flipped_rgb(
        rgb,
        2,
        2,
        false,
        BatchFormat::CNN,
        9,
        1.0
    );
    Batch rgb_batch = flipped_rgb.next();
    assert(rgb_batch.labels == std::vector<int>({0, 1}));
    assertClose(rgb_batch.images({0, 0, 0, 0}), 2.0);
    assertClose(rgb_batch.images({0, 0, 0, 2}), 0.0);
    assertClose(rgb_batch.images({0, 1, 0, 0}), 8.0);
    assertClose(rgb_batch.images({0, 2, 1, 0}), 17.0);
    assertClose(rgb_batch.images({1, 2, 1, 2}), 1015.0);

    Dataset reproducible_first = makeImageDataset({2, 3}, 8);
    Dataset reproducible_second = makeImageDataset({2, 3}, 8);
    DataLoader first(
        reproducible_first,
        3,
        8,
        true,
        BatchFormat::CNN,
        123,
        0.5
    );
    DataLoader second(
        reproducible_second,
        3,
        8,
        true,
        BatchFormat::CNN,
        123,
        0.5
    );

    while (first.hasNext() && second.hasNext()) {
        Batch first_batch = first.next();
        Batch second_batch = second.next();
        assert(first_batch.labels == second_batch.labels);
        assert(first_batch.images.getShape() == second_batch.images.getShape());
        for (int index = 0; index < first_batch.images.size(); index++) {
            assertClose(
                first_batch.images.flat(index),
                second_batch.images.flat(index)
            );
        }
    }
    assert(!first.hasNext());
    assert(!second.hasNext());
}
