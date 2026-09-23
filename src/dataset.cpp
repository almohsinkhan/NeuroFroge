#include "dataset.h" 
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>


Dataset::Dataset(const std::string& images_path,
                 const std::string& labels_path) {
    // Load images and labels from the specified paths
    // This is a placeholder implementation; actual loading logic should be added here.
    
    std::ifstream images_file(images_path);
    std::ifstream labels_file(labels_path);

    if (!images_file.is_open()) {
        throw std::runtime_error("Failed to open images file");
    }

    if (!labels_file.is_open()) {
        throw std::runtime_error("Failed to open labels file");
    }

    std::string line;
    while (std::getline(images_file, line)) {
        // Assuming each line in the images file represents a serialized Tensor
        std::istringstream iss(line);
        std::string value;

        Tensor image({784, 1});

        int i = 0;
        while (std::getline(iss, value, ',')) {
            if (i >= image.size()) {
                throw std::runtime_error("Image data exceeds expected size");
            }
            image.flat(i) = std::stod(value);
            i++;
        }
        if (i < image.size()) {
            throw std::runtime_error("Image data is incomplete");
        }

        images.push_back(image);
    }
    
    while (std::getline(labels_file, line)) {
        // Assuming each line in the labels file represents a label
        labels.push_back(std::stoi(line));
    }

    if (images.size() != labels.size()) {
        throw std::runtime_error("Number of images and labels do not match");
    }
}

Tensor Dataset::getImage(int index) const {
    if (index < 0 || index >= images.size()) {
        throw std::out_of_range("Index out of range");
    }
    return images[index];
}

int Dataset::getLabel(int index) const {
    if (index < 0 || index >= labels.size()) {
        throw std::out_of_range("Index out of range");
    }
    return labels[index];
}

size_t Dataset::size() const {
    return images.size();
}

