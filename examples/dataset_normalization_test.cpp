#include "dataset.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

void writeUint32(std::ofstream& file, std::uint32_t value) {
    char bytes[4] = {
        static_cast<char>((value >> 24) & 0xff),
        static_cast<char>((value >> 16) & 0xff),
        static_cast<char>((value >> 8) & 0xff),
        static_cast<char>(value & 0xff)
    };
    file.write(bytes, sizeof(bytes));
}

void writeCsv(
    const std::string& image_path,
    const std::string& label_path,
    const std::string& values
) {
    std::ofstream images(image_path);
    std::ofstream labels(label_path);
    images << values << '\n';
    labels << "0\n";
}

void writeIdx(
    const std::string& image_path,
    const std::string& label_path
) {
    std::ofstream images(image_path, std::ios::binary);
    std::ofstream labels(label_path, std::ios::binary);

    writeUint32(images, 2051);
    writeUint32(images, 1);
    writeUint32(images, 1);
    writeUint32(images, 2);
    images.put(static_cast<char>(0));
    images.put(static_cast<char>(255));

    writeUint32(labels, 2049);
    writeUint32(labels, 1);
    labels.put(static_cast<char>(0));
}

void assertNear(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

int main() {
    writeCsv(
        "/tmp/neurofroge_norm_csv_images",
        "/tmp/neurofroge_norm_csv_labels",
        "0,127.5,255"
    );

    Dataset csv_default(
        "/tmp/neurofroge_norm_csv_images",
        "/tmp/neurofroge_norm_csv_labels",
        {3}
    );
    Tensor default_image = csv_default.getImage(0);
    assertNear(default_image.flat(0), 0.0);
    assertNear(default_image.flat(1), 127.5);
    assertNear(default_image.flat(2), 255.0);

    Dataset csv_unit(
        "/tmp/neurofroge_norm_csv_images",
        "/tmp/neurofroge_norm_csv_labels",
        {3},
        Normalization::ScaleToUnit
    );
    Tensor unit_image = csv_unit.getImage(0);
    assertNear(unit_image.flat(0), 0.0);
    assertNear(unit_image.flat(1), 0.5);
    assertNear(unit_image.flat(2), 1.0);

    Dataset csv_standardized(
        "/tmp/neurofroge_norm_csv_images",
        "/tmp/neurofroge_norm_csv_labels",
        {3},
        Normalization::Standardize
    );
    Tensor standardized_image = csv_standardized.getImage(0);
    assertNear(standardized_image.flat(0), -1.224744871391589);
    assertNear(standardized_image.flat(1), 0.0);
    assertNear(standardized_image.flat(2), 1.224744871391589);

    writeCsv(
        "/tmp/neurofroge_norm_constant_images",
        "/tmp/neurofroge_norm_constant_labels",
        "4,4"
    );
    bool zero_variance_rejected = false;
    try {
        Dataset constant(
            "/tmp/neurofroge_norm_constant_images",
            "/tmp/neurofroge_norm_constant_labels",
            {2},
            Normalization::Standardize
        );
    } catch (const std::invalid_argument&) {
        zero_variance_rejected = true;
    }
    assert(zero_variance_rejected);

    writeIdx(
        "/tmp/neurofroge_norm_idx_images",
        "/tmp/neurofroge_norm_idx_labels"
    );

    Dataset idx_default = Dataset::fromIDX(
        "/tmp/neurofroge_norm_idx_images",
        "/tmp/neurofroge_norm_idx_labels"
    );
    assertNear(idx_default.getImage(0).flat(0), 0.0);
    assertNear(idx_default.getImage(0).flat(1), 1.0);

    Dataset idx_unit = Dataset::fromIDX(
        "/tmp/neurofroge_norm_idx_images",
        "/tmp/neurofroge_norm_idx_labels",
        Normalization::ScaleToUnit
    );
    assertNear(idx_unit.getImage(0).flat(0), 0.0);
    assertNear(idx_unit.getImage(0).flat(1), 1.0);

    Dataset idx_standardized = Dataset::fromIDX(
        "/tmp/neurofroge_norm_idx_images",
        "/tmp/neurofroge_norm_idx_labels",
        Normalization::Standardize
    );
    assertNear(idx_standardized.getImage(0).flat(0), -1.0);
    assertNear(idx_standardized.getImage(0).flat(1), 1.0);

    bool invalid_mode_rejected = false;
    try {
        Dataset invalid_mode(
            "/tmp/neurofroge_norm_csv_images",
            "/tmp/neurofroge_norm_csv_labels",
            {3},
            static_cast<Normalization>(99)
        );
    } catch (const std::invalid_argument&) {
        invalid_mode_rejected = true;
    }
    assert(invalid_mode_rejected);
}
