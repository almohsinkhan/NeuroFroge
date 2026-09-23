#pragma once

#include <string>
#include <vector>
#include "tensor.h"

class Dataset {
private:
    std::vector<Tensor> images;
    std::vector<int> labels;

public:
    Dataset(const std::string& imagePath,
            const std::string& labelPath);

    Tensor getImage(int index) const;
    int getLabel(int index) const;

    size_t size() const;
};