#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include <iostream>

int main() {

    Tensor a({2, 3});

    for (int i = 0; i < a.size(); i++)
        a.flat(i) = i + 1;

    Tensor b = transpose(a);

    std::cout << "A:\n";

    for (int i = 0; i < a.size(); i++) {
        std::cout << a.flat(i) << " ";

        if ((i + 1) % 3 == 0)
            std::cout << "\n";
    }

    std::cout << "\nA Transposed:\n";

    for (int i = 0; i < b.size(); i++) {
        std::cout << b.flat(i) << " ";

        if ((i + 1) % 2 == 0)
            std::cout << "\n";
    }

    b.flat(1) = 999;

    std::cout << "\nAfter modifying transpose:\n";

    for (int i = 0; i < a.size(); i++) {
        std::cout << a.flat(i) << " ";

        if ((i + 1) % 3 == 0)
            std::cout << "\n";
    }

    return 0;
}