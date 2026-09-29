#pragma once

#include <string>
#include <vector>
#include "tensor.h"

class Dataset {
private:
    std::vector<Tensor> images;
    std::vector<int> labels;
    int num_classes;

public:
    Dataset(const std::string& imagePath,
            const std::string& labelPath);

    Tensor getImage(int index) const;
    int getLabel(int index) const;
    int numClasses() const;
    size_t size() const;

};