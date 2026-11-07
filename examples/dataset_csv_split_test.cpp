#include "dataset.h"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void writeSplitFiles(
    const std::string& image_path,
    const std::string& label_path,
    int sample_count
) {
    std::ofstream images(image_path);
    std::ofstream labels(label_path);
    assert(images.is_open());
    assert(labels.is_open());

    for (int sample = 0; sample < sample_count; sample++) {
        images << sample << ',' << sample + 100 << '\n';
        labels << sample << '\n';
    }
}

void assertPairing(const Dataset& dataset) {
    for (int index = 0; index < static_cast<int>(dataset.size()); index++) {
        assert(
            dataset.getImage(index).flat(0) ==
            static_cast<double>(dataset.getLabel(index))
        );
        assert(
            dataset.getImage(index).flat(1) ==
            static_cast<double>(dataset.getLabel(index) + 100)
        );
    }
}

int main() {
    const std::string image_path = "/tmp/neurofroge_split_images.csv";
    const std::string label_path = "/tmp/neurofroge_split_labels.csv";
    writeSplitFiles(image_path, label_path, 10);

    Dataset dataset(image_path, label_path, {2});
    auto first_split = dataset.splitCsv(0.3, 1234);
    auto second_split = dataset.splitCsv(0.3, 1234);

    const Dataset& train = first_split.first;
    const Dataset& validation = first_split.second;
    assert(train.size() == 7);
    assert(validation.size() == 3);
    assert(train.sampleShape() == std::vector<int>({2}));
    assert(validation.sampleShape() == std::vector<int>({2}));
    assertPairing(train);
    assertPairing(validation);

    for (int index = 0; index < static_cast<int>(train.size()); index++) {
        assert(
            train.getLabel(index) ==
            second_split.first.getLabel(index)
        );
    }
    for (int index = 0;
         index < static_cast<int>(validation.size());
         index++) {
        assert(
            validation.getLabel(index) ==
            second_split.second.getLabel(index)
        );
    }

    Dataset unsplit(image_path, label_path, {2});
    assert(unsplit.size() == 10);
    assert(unsplit.getLabel(0) == 0);
    assert(unsplit.getImage(0).flat(0) == 0.0);

    for (double invalid_ratio : {0.0, -0.1, 1.0, 1.1}) {
        bool rejected = false;
        try {
            dataset.splitCsv(invalid_ratio, 1234);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    bool empty_split_rejected = false;
    try {
        dataset.splitCsv(0.01, 1234);
    } catch (const std::invalid_argument&) {
        empty_split_rejected = true;
    }
    assert(empty_split_rejected);

    std::ofstream empty_images("/tmp/neurofroge_empty_images.csv");
    std::ofstream empty_labels("/tmp/neurofroge_empty_labels.csv");
    empty_images.close();
    empty_labels.close();
    Dataset empty(
        "/tmp/neurofroge_empty_images.csv",
        "/tmp/neurofroge_empty_labels.csv",
        {2}
    );

    bool empty_dataset_rejected = false;
    try {
        empty.splitCsv(0.3, 1234);
    } catch (const std::invalid_argument&) {
        empty_dataset_rejected = true;
    }
    assert(empty_dataset_rejected);
}
