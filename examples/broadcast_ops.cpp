#include "../include/tensor_ops.h"
#include <iostream>

void printTensor(const Tensor& T) {

    for (int i = 0; i < T.size(); i++) {
        std::cout << T.flat(i) << " ";

        if ((i + 1) % T.getShape().back() == 0)
            std::cout << "\n";
    }

    std::cout << "\n";
}

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

    std::cout << "Add:\n";
    printTensor(add(A, B));

    std::cout << "Multiply:\n";
    printTensor(multiply(A, B));

    std::cout << "Subtract:\n";
    printTensor(subtract(A, B));

    return 0;
}