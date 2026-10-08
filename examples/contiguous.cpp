#include "../include/tensor.h"
#include <iostream>

int main() {

    Tensor a({3, 1});
    Tensor b = a.broadcastTo({3, 4});

    std::cout << "A contiguous: "
              << a.isContiguous() << "\n";

    std::cout << "B contiguous: "
              << b.isContiguous() << "\n";

    return 0;
}