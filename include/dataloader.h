#pragma once
#include <vector>
#include "dataset.h"

struct Batch {
    Tensor images;
    std::vector<int> labels;
};

class DataLoader {
private:
    Dataset& dataset;
    int batch_size;
    int max_samples;
    int current_index;

public:
    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples
    );

    bool hasNext() const;
    Batch next();
    void reset();
};