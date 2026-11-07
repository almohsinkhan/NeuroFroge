#pragma once

#include "dataset.h"
#include "tensor.h"
#include <cstdint>
#include <random>
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
    double horizontal_flip_probability;

    std::vector<int> indices;
    std::mt19937 generator;

    void initializeIndices();

public:
    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle = true,
        BatchFormat format = BatchFormat::MLP
    );

    DataLoader(
        Dataset& dataset,
        int batch_size,
        int max_samples,
        bool shuffle,
        BatchFormat format,
        std::uint32_t seed,
        double horizontal_flip_probability = 0.0
    );

    bool hasNext() const;

    Batch next();

    void reset();
};