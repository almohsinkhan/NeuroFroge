#include "idx_reader.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

std::uint32_t readBigEndianUint32(
    std::ifstream& file,
    const std::string& path
) {
    std::uint8_t bytes[4];

    if (!file.read(
            reinterpret_cast<char*>(bytes),
            sizeof(bytes)
        )) {
        throw std::runtime_error(
            "Incomplete IDX header in file: " + path
        );
    }

    return
        (static_cast<std::uint32_t>(bytes[0]) << 24) |
        (static_cast<std::uint32_t>(bytes[1]) << 16) |
        (static_cast<std::uint32_t>(bytes[2]) << 8) |
        static_cast<std::uint32_t>(bytes[3]);
}

void ensureNoTrailingData(
    std::ifstream& file,
    const std::string& path
) {
    char extra;
    if (file.read(&extra, 1)) {
        throw std::runtime_error(
            "Unexpected trailing data in IDX file: " + path
        );
    }

    if (!file.eof()) {
        throw std::runtime_error(
            "Failed while checking IDX file: " + path
        );
    }
}

}

IDXReader::IDXReader(
    const std::string& image_path,
    const std::string& label_path
)
    : image_count(0),
      image_height(0),
      image_width(0)
{
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        throw std::runtime_error(
            "Failed to open IDX image file: " + image_path
        );
    }

    std::uint32_t image_magic =
        readBigEndianUint32(image_file, image_path);
    if (image_magic != 2051) {
        throw std::runtime_error(
            "Invalid IDX image magic number in file: " + image_path
        );
    }

    std::uint32_t declared_image_count =
        readBigEndianUint32(image_file, image_path);
    std::uint32_t declared_height =
        readBigEndianUint32(image_file, image_path);
    std::uint32_t declared_width =
        readBigEndianUint32(image_file, image_path);

    if (declared_image_count == 0) {
        throw std::runtime_error(
            "IDX image file contains no images: " + image_path
        );
    }

    if (declared_height != 28 || declared_width != 28) {
        throw std::runtime_error(
            "IDX image dimensions must be 28x28: " + image_path
        );
    }

    const std::size_t pixels_per_image =
        static_cast<std::size_t>(declared_height) *
        static_cast<std::size_t>(declared_width);

    if (static_cast<std::size_t>(declared_image_count) >
        std::numeric_limits<std::size_t>::max() /
        pixels_per_image) {
        throw std::runtime_error(
            "IDX image dimensions overflow memory size: " + image_path
        );
    }

    image_count = declared_image_count;
    image_height = declared_height;
    image_width = declared_width;

    image_data.resize(image_count * pixels_per_image);
    if (!image_file.read(
            reinterpret_cast<char*>(image_data.data()),
            static_cast<std::streamsize>(image_data.size())
        )) {
        throw std::runtime_error(
            "Incomplete IDX image data in file: " + image_path
        );
    }

    ensureNoTrailingData(image_file, image_path);

    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        throw std::runtime_error(
            "Failed to open IDX label file: " + label_path
        );
    }

    std::uint32_t label_magic =
        readBigEndianUint32(label_file, label_path);
    if (label_magic != 2049) {
        throw std::runtime_error(
            "Invalid IDX label magic number in file: " + label_path
        );
    }

    std::uint32_t declared_label_count =
        readBigEndianUint32(label_file, label_path);
    if (declared_label_count != declared_image_count) {
        throw std::runtime_error(
            "IDX image and label counts do not match"
        );
    }

    label_data.resize(declared_label_count);
    if (!label_file.read(
            reinterpret_cast<char*>(label_data.data()),
            static_cast<std::streamsize>(label_data.size())
        )) {
        throw std::runtime_error(
            "Incomplete IDX label data in file: " + label_path
        );
    }

    for (std::uint8_t label : label_data) {
        if (label > 9) {
            throw std::runtime_error(
                "IDX label must be in the range 0-9: " + label_path
            );
        }
    }

    ensureNoTrailingData(label_file, label_path);
}

const std::vector<std::uint8_t>& IDXReader::images() const {
    return image_data;
}

const std::vector<std::uint8_t>& IDXReader::labels() const {
    return label_data;
}

std::size_t IDXReader::count() const {
    return image_count;
}

std::size_t IDXReader::height() const {
    return image_height;
}

std::size_t IDXReader::width() const {
    return image_width;
}
