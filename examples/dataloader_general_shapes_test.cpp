#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <stdexcept>
#include <vector>

Dataset makeDataset(const std::vector<std::vector<int>>& shapes) {
    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;

    for (std::size_t sample = 0; sample < shapes.size(); sample++) {
        Tensor image(shapes[sample]);
        for (int i = 0; i < image.size(); i++) {
            image.flat(i) =
                static_cast<double>(sample * image.size() + i);
        }
        dataset.images.push_back(image);
        dataset.labels.push_back(static_cast<int>(sample));
    }

    return dataset;
}

void assertShape(
    const Tensor& tensor,
    const std::vector<int>& expected
) {
    assert(tensor.getShape() == expected);
}

int main() {
    Dataset mlp_dataset = makeDataset({
        {100},
        {100},
        {100}
    });
    DataLoader mlp_loader(
        mlp_dataset,
        2,
        3,
        false,
        BatchFormat::MLP
    );
    Batch mlp_batch = mlp_loader.next();
    assertShape(mlp_batch.images, {100, 2});
    assert(mlp_batch.labels == std::vector<int>({0, 1}));
    assert(mlp_batch.images({0, 0}) == 0.0);
    assert(mlp_batch.images({99, 1}) == 199.0);

    Dataset grayscale_3d_dataset = makeDataset({
        {1, 32, 32},
        {1, 32, 32}
    });
    DataLoader grayscale_3d_loader(
        grayscale_3d_dataset,
        2,
        2,
        false,
        BatchFormat::CNN
    );
    Batch grayscale_3d_batch = grayscale_3d_loader.next();
    assertShape(
        grayscale_3d_batch.images,
        {2, 1, 32, 32}
    );
    assert(
        grayscale_3d_batch.images({1, 0, 31, 31}) ==
        2047.0
    );

    Dataset grayscale_2d_dataset = makeDataset({
        {32, 32},
        {32, 32}
    });
    DataLoader grayscale_2d_loader(
        grayscale_2d_dataset,
        2,
        2,
        false,
        BatchFormat::CNN
    );
    Batch grayscale_2d_batch = grayscale_2d_loader.next();
    assertShape(
        grayscale_2d_batch.images,
        {2, 1, 32, 32}
    );
    assert(
        grayscale_2d_batch.images({0, 0, 31, 31}) ==
        1023.0
    );

    Dataset rgb_dataset = makeDataset({
        {3, 32, 32},
        {3, 32, 32}
    });
    DataLoader rgb_loader(
        rgb_dataset,
        2,
        2,
        false,
        BatchFormat::CNN
    );
    Batch rgb_batch = rgb_loader.next();
    assertShape(rgb_batch.images, {2, 3, 32, 32});
    assert(rgb_batch.images({1, 2, 31, 31}) == 6143.0);

    Dataset inconsistent_dataset = makeDataset({
        {32, 32},
        {3, 32, 32}
    });
    DataLoader inconsistent_loader(
        inconsistent_dataset,
        2,
        2,
        false,
        BatchFormat::CNN
    );

    bool rejected = false;
    try {
        inconsistent_loader.next();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    Dataset fashion_dataset = Dataset::fromIDX(
        "data/FashionMNIST/raw/train-images-idx3-ubyte",
        "data/FashionMNIST/raw/train-labels-idx1-ubyte"
    );

    DataLoader fashion_mlp_loader(
        fashion_dataset,
        2,
        2,
        false,
        BatchFormat::MLP
    );
    Batch fashion_mlp_batch = fashion_mlp_loader.next();
    assertShape(fashion_mlp_batch.images, {784, 2});

    DataLoader fashion_cnn_loader(
        fashion_dataset,
        2,
        2,
        false,
        BatchFormat::CNN
    );
    Batch fashion_cnn_batch = fashion_cnn_loader.next();
    assertShape(
        fashion_cnn_batch.images,
        {2, 1, 28, 28}
    );
}
