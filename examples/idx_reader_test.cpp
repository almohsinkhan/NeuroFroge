#include "idx_reader.h"

#include <cassert>
#include <iostream>

int main() {
    IDXReader reader(
        "data/FashionMNIST/raw/train-images-idx3-ubyte",
        "data/FashionMNIST/raw/train-labels-idx1-ubyte"
    );

    assert(reader.count() == 60000);
    assert(reader.labels().size() == 60000);
    assert(reader.height() == 28);
    assert(reader.width() == 28);
    assert(reader.images().size() == 60000 * 28 * 28);

    for (std::uint8_t label : reader.labels()) {
        assert(label <= 9);
    }

    std::cout << "First label: "
              << static_cast<int>(reader.labels().front())
              << '\n';

    std::cout << "First image pixels: ";
    for (int i = 0; i < 10; i++) {
        std::cout << static_cast<int>(reader.images()[i]);
        if (i < 9) {
            std::cout << ", ";
        }
    }
    std::cout << '\n';
}
