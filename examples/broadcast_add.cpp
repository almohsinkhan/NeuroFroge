#include "../include/tensor_ops.h"
#include <iostream>

int main() {

    Tensor A({3, 1});
    Tensor B({1, 4});

    A.flat(0) = 10;
    A.flat(1) = 20;
    A.flat(2) = 30;

    B.flat(0) = 1;
    B.flat(1) = 2;
    B.flat(2) = 3;
    B.flat(3) = 4;

    Tensor C = add(A, B);

    for (int i = 0; i < C.size(); i++) {
        std::cout << C.flat(i) << " ";

        if ((i + 1) % 4 == 0)
            std::cout << "\n";
    }

    return 0;
}