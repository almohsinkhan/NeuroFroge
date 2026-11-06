#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include "tensor.h"

class Dataset {
private:
    std::vector<Tensor> images;
    std::vector<std::uint8_t> image_pixels;
    std::vector<int> labels;
    bool uses_idx;
    int image_height;
    int image_width;
    int image_channels;

    Dataset();

public:
    Dataset(const std::string& imagePath,
            const std::string& labelPath);

    static Dataset fromIDX(
        const std::string& imagePath,
        const std::string& labelPath
    );

    Tensor getImage(int index) const;
    int getLabel(int index) const;
    int numClasses() const;
    size_t size() const;

    std::vector<int> sampleShape() const;

};