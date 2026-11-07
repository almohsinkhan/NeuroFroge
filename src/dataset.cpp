#include "dataset.h" 
#include "idx_reader.h"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>

Dataset::Dataset()
    : uses_idx(false),
      image_height(28),
      image_width(28),
      image_channels(1)
{
}

Dataset::Dataset(const std::string& images_path,
                 const std::string& labels_path)
    : Dataset(images_path, labels_path, {28, 28}) {}

Dataset::Dataset(
    const std::string& images_path,
    const std::string& labels_path,
    const std::vector<int>& sample_shape
) {
    uses_idx = false;
    image_height = 28;
    image_width = 28;
    image_channels = 1;

    if (sample_shape.empty()) {
        throw std::invalid_argument(
            "CSV sample shape must not be empty"
        );
    }

    std::size_t sample_size = 1;
    for (int dimension : sample_shape) {
        if (dimension <= 0 ||
            sample_size >
                std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimension)) {
            throw std::invalid_argument(
                "CSV sample shape must contain valid dimensions"
            );
        }
        sample_size *= static_cast<std::size_t>(dimension);
    }

    if (sample_size > static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )) {
        throw std::invalid_argument(
            "CSV sample shape is too large"
        );
    }

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
        std::istringstream iss(line);
        std::string value;

        Tensor image(sample_shape);

        int i = 0;
        while (std::getline(iss, value, ',')) {
            if (i >= image.size()) {
                throw std::runtime_error(
                    "CSV row contains more values than its sample shape"
                );
            }
            image.flat(i) = std::stod(value);
            i++;
        }
        if (i < image.size()) {
            throw std::runtime_error(
                "CSV row contains fewer values than its sample shape"
            );
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

Dataset Dataset::fromIDX(
    const std::string& image_path,
    const std::string& label_path
) {
    IDXReader reader(image_path, label_path);
    Dataset dataset;

    dataset.uses_idx = true;
    dataset.image_height = static_cast<int>(reader.height());
    dataset.image_width = static_cast<int>(reader.width());
    dataset.image_channels = 1;
    dataset.image_pixels = reader.images();

    dataset.labels.reserve(reader.labels().size());
    for (std::uint8_t label : reader.labels()) {
        dataset.labels.push_back(static_cast<int>(label));
    }

    return dataset;
}

Tensor Dataset::getImage(int index) const {
    if (index < 0 || index >= static_cast<int>(size())) {
        throw std::out_of_range("Index out of range");
    }

    if (uses_idx) {
        Tensor image({
            image_channels,
            image_height,
            image_width
        });

        int image_size =
            image_channels * image_height * image_width;
        int offset = index * image_size;

        for (int i = 0; i < image_size; i++) {
            image.flat(i) =
                static_cast<double>(image_pixels[offset + i]) / 255.0;
        }

        return image;
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
    return uses_idx
        ? labels.size()
        : images.size();
}

int Dataset::numClasses() const {
    std::unordered_set<int> unique_labels(
        labels.begin(),
        labels.end()
    );

    return unique_labels.size();
}

std::vector<int> Dataset::sampleShape() const {
    if (size() == 0) {
        throw std::runtime_error("Dataset is empty");
    }

    if (uses_idx) {
        return {
            image_channels,
            image_height,
            image_width
        };
    }

    return images[0].getShape();
}
