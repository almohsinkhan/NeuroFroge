#include <iostream>
#include <chrono>

#include "tensor.h"
#include "tensor_ops.h"

int main() {

    // -------------------------
    // Correctness test
    // -------------------------

    Tensor A({5, 4});
    Tensor B({5, 3});

    for (int i = 0; i < A.size(); i++)
        A.flat(i) = i + 1;

    for (int i = 0; i < B.size(); i++)
        B.flat(i) = i + 1;

    Tensor A_T = transpose(A);

    Tensor C1 = matmul(A_T, B);
    Tensor C2 = matmul_transpose_left(A, B);

    std::cout << "Correctness test:\n";

    for (int i = 0; i < C1.size(); i++) {
        std::cout << C1.flat(i)
                  << "  "
                  << C2.flat(i)
                  << "\n";
    }


    // -------------------------
    // Matmul benchmark
    // -------------------------

    Tensor X({128, 784});
    Tensor Y({784, 128});

    for (int i = 0; i < X.size(); i++)
        X.flat(i) = 0.5;

    for (int i = 0; i < Y.size(); i++)
        Y.flat(i) = 0.2;

    const int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        Tensor result = matmul(X, Y);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "\nMatmul benchmark:\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Time: " << seconds << " seconds\n";

    return 0;
}