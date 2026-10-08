#pragma once

#include "dataset.h"
#include "tensor.h"
#include <vector>

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

    bool shuffle;

    std::vector<int> indices;

    void initializeIndices();

public:
    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle = true
    );

    bool hasNext() const;

    Batch next();

    void reset();
};