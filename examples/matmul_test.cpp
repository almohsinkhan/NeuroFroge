#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include <iostream>

int main() {

    Tensor A({2, 3});

    // A =
    // 1 2 3
    // 4 5 6
    A.flat(0) = 1;
    A.flat(1) = 2;
    A.flat(2) = 3;
    A.flat(3) = 4;
    A.flat(4) = 5;
    A.flat(5) = 6;

    Tensor B2({2, 3});

B2.flat(0) = 7;
B2.flat(1) = 8;
B2.flat(2) = 9;
B2.flat(3) = 10;
B2.flat(4) = 11;
B2.flat(5) = 12;

Tensor E = matmul_transpose_left(B2, A);

std::cout << "\nB2^T x A:\n";

for (int i = 0; i < E.size(); i++) {

    std::cout << E.flat(i) << " ";

    if ((i + 1) % 3 == 0)
        std::cout << "\n";
}

    return 0;
}