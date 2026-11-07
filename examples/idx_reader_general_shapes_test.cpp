#include "dataset.h"
#include "idx_reader.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
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

void writeIDX(
    const std::string& image_path,
    const std::string& label_path,
    std::uint32_t image_magic,
    std::uint32_t image_count,
    std::uint32_t height,
    std::uint32_t width,
    std::uint32_t label_count,
    bool truncate_images,
    bool truncate_labels
) {
    std::ofstream images(image_path, std::ios::binary);
    std::ofstream labels(label_path, std::ios::binary);
    assert(images.is_open());
    assert(labels.is_open());

    writeUint32(images, image_magic);
    writeUint32(images, image_count);
    writeUint32(images, height);
    writeUint32(images, width);

    const std::size_t pixel_count =
        static_cast<std::size_t>(image_count) *
        height *
        width;
    for (std::size_t i = 0; i < pixel_count; i++) {
        if (truncate_images && i + 1 == pixel_count) {
            break;
        }
        images.put(static_cast<char>(i % 256));
    }

    writeUint32(labels, 2049);
    writeUint32(labels, label_count);
    for (std::uint32_t i = 0; i < label_count; i++) {
        if (truncate_labels && i + 1 == label_count) {
            break;
        }
        labels.put(static_cast<char>(i % 10));
    }
}

void assertRejected(
    const std::function<void()>& operation
) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

int main() {
    writeIDX(
        "/tmp/neurofroge_idx_16_images",
        "/tmp/neurofroge_idx_16_labels",
        2051,
        2,
        16,
        16,
        2,
        false,
        false
    );
    IDXReader reader16(
        "/tmp/neurofroge_idx_16_images",
        "/tmp/neurofroge_idx_16_labels"
    );
    assert(reader16.count() == 2);
    assert(reader16.height() == 16);
    assert(reader16.width() == 16);
    assert(reader16.images().size() == 2 * 16 * 16);

    Dataset dataset16 = Dataset::fromIDX(
        "/tmp/neurofroge_idx_16_images",
        "/tmp/neurofroge_idx_16_labels"
    );
    assert(dataset16.sampleShape() ==
           std::vector<int>({1, 16, 16}));
    assert(dataset16.getImage(0).size() == 16 * 16);

    writeIDX(
        "/tmp/neurofroge_idx_32_images",
        "/tmp/neurofroge_idx_32_labels",
        2051,
        1,
        32,
        32,
        1,
        false,
        false
    );
    IDXReader reader32(
        "/tmp/neurofroge_idx_32_images",
        "/tmp/neurofroge_idx_32_labels"
    );
    assert(reader32.height() == 32);
    assert(reader32.width() == 32);
    assert(reader32.images().size() == 32 * 32);

    assertRejected([] {
        writeIDX(
            "/tmp/neurofroge_idx_bad_magic_images",
            "/tmp/neurofroge_idx_bad_magic_labels",
            9999,
            1,
            16,
            16,
            1,
            false,
            false
        );
        IDXReader reader(
            "/tmp/neurofroge_idx_bad_magic_images",
            "/tmp/neurofroge_idx_bad_magic_labels"
        );
    });

    assertRejected([] {
        std::ofstream image_file(
            "/tmp/neurofroge_idx_truncated_header",
            std::ios::binary
        );
        writeUint32(image_file, 2051);
        IDXReader reader(
            "/tmp/neurofroge_idx_truncated_header",
            "/tmp/neurofroge_idx_16_labels"
        );
    });

    assertRejected([] {
        writeIDX(
            "/tmp/neurofroge_idx_truncated_images",
            "/tmp/neurofroge_idx_truncated_image_labels",
            2051,
            1,
            16,
            16,
            1,
            true,
            false
        );
        IDXReader reader(
            "/tmp/neurofroge_idx_truncated_images",
            "/tmp/neurofroge_idx_truncated_image_labels"
        );
    });

    assertRejected([] {
        writeIDX(
            "/tmp/neurofroge_idx_zero_dimension_images",
            "/tmp/neurofroge_idx_zero_dimension_labels",
            2051,
            1,
            0,
            16,
            1,
            false,
            false
        );
        IDXReader reader(
            "/tmp/neurofroge_idx_zero_dimension_images",
            "/tmp/neurofroge_idx_zero_dimension_labels"
        );
    });

    assertRejected([] {
        writeIDX(
            "/tmp/neurofroge_idx_mismatched_images",
            "/tmp/neurofroge_idx_mismatched_labels",
            2051,
            2,
            16,
            16,
            1,
            false,
            false
        );
        IDXReader reader(
            "/tmp/neurofroge_idx_mismatched_images",
            "/tmp/neurofroge_idx_mismatched_labels"
        );
    });
}
