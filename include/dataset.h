#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include "tensor.h"

enum class Normalization {
    None,
    ScaleToUnit,
    Standardize
};

class Dataset {
private:
    std::vector<Tensor> images;
    std::vector<std::uint8_t> image_pixels;
    std::vector<int> labels;
    bool uses_idx;
    int image_height;
    int image_width;
    int image_channels;
    Normalization normalization;
    double normalization_mean;
    double normalization_stddev;

    Dataset();

    void calculateNormalizationStatistics();
    Tensor applyNormalization(Tensor image) const;

public:
    Dataset(const std::string& imagePath,
            const std::string& labelPath);

    Dataset(
        const std::string& imagePath,
        const std::string& labelPath,
        const std::vector<int>& sample_shape,
        Normalization normalization_mode = Normalization::None
    );

    static Dataset fromIDX(
        const std::string& imagePath,
        const std::string& labelPath,
        Normalization normalization = Normalization::ScaleToUnit
    );

    Tensor getImage(int index) const;
    int getLabel(int index) const;
    int numClasses() const;
    size_t size() const;

    std::vector<int> sampleShape() const;

};