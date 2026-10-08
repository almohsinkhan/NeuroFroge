#include "../include/tensor.h"
#include <iostream>

int main() {

    Tensor a({3, 4});

    a.flat(0) = 10;
    a.flat(1) = 20;
    a.flat(2) = 30;

    Tensor b = a.broadcastTo({3, 4});

    for (int i = 0; i < b.size(); i++) {
        std::cout << b.flat(i) << " ";

        if ((i + 1) % 4 == 0)
            std::cout << "\n";
    }

    return 0;
}