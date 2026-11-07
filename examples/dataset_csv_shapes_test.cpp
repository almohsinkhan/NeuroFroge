#include "dataset.h"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void writeCsvFiles(
    const std::string& image_path,
    const std::string& label_path,
    int rows,
    int values_per_row
) {
    std::ofstream images(image_path);
    std::ofstream labels(label_path);
    assert(images.is_open());
    assert(labels.is_open());

    for (int row = 0; row < rows; row++) {
        for (int value = 0; value < values_per_row; value++) {
            if (value > 0) {
                images << ',';
            }
            images << row * values_per_row + value;
        }
        images << '\n';
        labels << row % 3 << '\n';
    }
}

void assertShape(
    const Dataset& dataset,
    const std::vector<int>& expected
) {
    assert(dataset.sampleShape() == expected);
}

int main() {
    writeCsvFiles(
        "/tmp/neurofroge_csv_100_images.csv",
        "/tmp/neurofroge_csv_100_labels.csv",
        2,
        100
    );
    Dataset features(
        "/tmp/neurofroge_csv_100_images.csv",
        "/tmp/neurofroge_csv_100_labels.csv",
        {100}
    );
    assertShape(features, {100});
    assert(features.size() == 2);
    assert(features.getImage(1).flat(99) == 199.0);
    assert(features.getLabel(1) == 1);

    writeCsvFiles(
        "/tmp/neurofroge_csv_gray_images.csv",
        "/tmp/neurofroge_csv_gray_labels.csv",
        2,
        32 * 32
    );
    Dataset grayscale(
        "/tmp/neurofroge_csv_gray_images.csv",
        "/tmp/neurofroge_csv_gray_labels.csv",
        {32, 32}
    );
    assertShape(grayscale, {32, 32});
    assert(grayscale.getImage(1).getShape() == std::vector<int>({
        32,
        32
    }));

    writeCsvFiles(
        "/tmp/neurofroge_csv_rgb_images.csv",
        "/tmp/neurofroge_csv_rgb_labels.csv",
        2,
        3 * 32 * 32
    );
    Dataset rgb(
        "/tmp/neurofroge_csv_rgb_images.csv",
        "/tmp/neurofroge_csv_rgb_labels.csv",
        {3, 32, 32}
    );
    assertShape(rgb, {3, 32, 32});
    assert(rgb.getImage(1).flat(3 * 32 * 32 - 1) == 6143.0);

    writeCsvFiles(
        "/tmp/neurofroge_csv_invalid_images.csv",
        "/tmp/neurofroge_csv_invalid_labels.csv",
        1,
        99
    );

    bool rejected = false;
    try {
        Dataset invalid(
            "/tmp/neurofroge_csv_invalid_images.csv",
            "/tmp/neurofroge_csv_invalid_labels.csv",
            {100}
        );
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    Dataset fashion(
        "data/images.csv",
        "data/labels.csv"
    );
    assertShape(fashion, {28, 28});
    assert(fashion.size() == 1000);
    assert(fashion.getImage(0).size() == 784);
}
