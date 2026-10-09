#include "tensor.h"
#include "tensor_ops.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool close(double a, double b) {
    return std::abs(a - b) < 1e-9;
}

int main() {
    // A: (2, 3)
    Tensor A({2, 3});
    double a[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
        A.flat(i) = a[i];

    // B: (2, 3)
    Tensor B({2, 3});
    double b[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++)
        B.flat(i) = b[i];

    // A * B^T -> (2, 2)
    Tensor right = matmul_transpose_right(A, B);

    assert(right.getShape() == std::vector<int>({2, 2}));
    assert(close(right.flat(0), 50));
    assert(close(right.flat(1), 68));
    assert(close(right.flat(2), 122));
    assert(close(right.flat(3), 167));

    // A^T * B -> (3, 3)
    Tensor left = matmul_transpose_left(A, B);

    assert(left.getShape() == std::vector<int>({3, 3}));
    assert(close(left.flat(0), 47));
    assert(close(left.flat(1), 52));
    assert(close(left.flat(2), 57));
    assert(close(left.flat(3), 64));
    assert(close(left.flat(4), 71));
    assert(close(left.flat(5), 78));
    assert(close(left.flat(6), 81));
    assert(close(left.flat(7), 90));
    assert(close(left.flat(8), 99));

    std::cout << "Both optimized matmul tests passed!\n";
    return 0;
}