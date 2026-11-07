#define private public
#include "dataset.h"
#undef private

#include "dataloader.h"

#include <cassert>
#include <set>
#include <vector>

Dataset makeDataset(int sample_count) {
    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    dataset.images.clear();
    dataset.labels.clear();
    dataset.uses_idx = false;

    for (int sample = 0; sample < sample_count; sample++) {
        Tensor image({2});
        image.flat(0) = sample;
        image.flat(1) = sample + 100;
        dataset.images.push_back(image);
        dataset.labels.push_back(sample);
    }

    return dataset;
}

std::vector<int> readEpoch(
    DataLoader& loader,
    int expected_size
) {
    std::vector<int> order;
    std::set<int> seen;

    while (loader.hasNext()) {
        Batch batch = loader.next();
        assert(batch.images.getShape()[1] == static_cast<int>(
            batch.labels.size()
        ));

        for (int sample = 0;
             sample < static_cast<int>(batch.labels.size());
             sample++) {
            int label = batch.labels[sample];
            assert(batch.images({0, sample}) == label);
            assert(batch.images({1, sample}) == label + 100);
            assert(seen.insert(label).second);
            order.push_back(label);
        }
    }

    assert(order.size() == static_cast<std::size_t>(expected_size));
    assert(seen.size() == static_cast<std::size_t>(expected_size));
    return order;
}

int main() {
    Dataset first_dataset = makeDataset(11);
    Dataset second_dataset = makeDataset(11);
    DataLoader first(
        first_dataset,
        3,
        11,
        true,
        BatchFormat::MLP,
        12345
    );
    DataLoader second(
        second_dataset,
        3,
        11,
        true,
        BatchFormat::MLP,
        12345
    );

    std::vector<int> first_epoch = readEpoch(first, 11);
    std::vector<int> second_epoch = readEpoch(second, 11);
    assert(first_epoch == second_epoch);

    first.reset();
    second.reset();
    first_epoch = readEpoch(first, 11);
    second_epoch = readEpoch(second, 11);
    assert(first_epoch == second_epoch);

    Dataset different_dataset = makeDataset(11);
    DataLoader different(
        different_dataset,
        3,
        11,
        true,
        BatchFormat::MLP,
        54321
    );
    assert(first_epoch != readEpoch(different, 11));

    Dataset no_shuffle_dataset = makeDataset(11);
    DataLoader no_shuffle(
        no_shuffle_dataset,
        4,
        11,
        false,
        BatchFormat::MLP,
        12345
    );
    std::vector<int> ordered = readEpoch(no_shuffle, 11);
    for (int index = 0; index < 11; index++) {
        assert(ordered[index] == index);
    }

    no_shuffle.reset();
    ordered = readEpoch(no_shuffle, 11);
    for (int index = 0; index < 11; index++) {
        assert(ordered[index] == index);
    }
}
