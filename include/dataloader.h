#pragma once

#include "dataset.h"
#include "tensor.h"
#include <vector>

struct Batch {
    Tensor images;
    std::vector<int> labels;
};

enum class BatchFormat {
    MLP,
    CNN
};

class DataLoader {

private:
    Dataset& dataset;

    int batch_size;
    int max_samples;
    int current_index;

    bool shuffle;
    BatchFormat format;

    std::vector<int> indices;

    void initializeIndices();

public:
    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle = true,
        BatchFormat format = BatchFormat::MLP
    );

    bool hasNext() const;

    Batch next();

    void reset();
};