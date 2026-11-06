#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


class IDXReader {
private:
    std::vector<std::uint8_t> image_data;
    std::vector<std::uint8_t> label_data;
    std::size_t image_count;
    std::size_t image_height;
    std::size_t image_width;

public:
    IDXReader(
        const std::string& image_path,
        const std::string& label_path
    );

    const std::vector<std::uint8_t>& images() const;
    const std::vector<std::uint8_t>& labels() const;

    std::size_t count() const;
    std::size_t height() const;
    std::size_t width() const;
};
