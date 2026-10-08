#include "tensor_ops.h"

#include <cassert>
#include <vector>
#include <algorithm>

std::vector<int> broadcastShape(
    const Tensor& A,
    const Tensor& B
) {
    int ndimA = A.ndim();
    int ndimB = B.ndim();

    int resultNdim = std::max(ndimA, ndimB);

    std::vector<int> resultShape(resultNdim);

    for (int i = 0; i < resultNdim; i++) {

        int dimA = (i < ndimA)
            ? A.getShape()[ndimA - 1 - i]
            : 1;

        int dimB = (i < ndimB)
            ? B.getShape()[ndimB - 1 - i]
            : 1;

        assert(dimA == dimB || dimA == 1 || dimB == 1);

        resultShape[resultNdim - 1 - i] =
            std::max(dimA, dimB);
    }

    return resultShape;
}


template <typename Operation>
Tensor elementwise(
    const Tensor& A,
    const Tensor& B,
    Operation op
) {
    std::vector<int> resultShape = broadcastShape(A, B);

    Tensor A_broadcasted = A.broadcastTo(resultShape);
    Tensor B_broadcasted = B.broadcastTo(resultShape);

    Tensor C(resultShape);

    for (int i = 0; i < C.size(); i++) {
        C.flat(i) = op(
            A_broadcasted.flat(i),
            B_broadcasted.flat(i)
        );
    }

    return C;
}

Tensor matmul(const Tensor& A, const Tensor& B) {

    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.getShape()[1] == B.getShape()[0]);

    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int p = B.getShape()[1];

    Tensor C({m, p});

    /*OLD: i → j → k
      NEW: i → k → j
      which is better for cache locality (contiguous in memory)*/
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++){

            double a = A.flat(i * n + k);

            for (int j = 0; j < p; j++) {
                C.flat(i * p + j) += a * B.flat(k * p + j);
            }
        }
    }

    return C;
}

Tensor matmul_transpose_right(
    const Tensor& A,
    const Tensor& B
) {
    assert(A.ndim() == 2 && B.ndim() == 2);

    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int p = B.getShape()[0];

    assert(n == B.getShape()[1]);

    Tensor C({m, p});

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A.flat(i * n + k)
                     * B.flat(j * n + k);
            }

            C.flat(i * p + j) = sum;
        }
    }

    return C;
}

Tensor matmul_transpose_left(
    const Tensor& A,
    const Tensor& B
) {
    assert(A.ndim() == 2 && B.ndim() == 2);

    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int p = B.getShape()[1];

    // A^T is n × m, so B must be m × p
    assert(m == B.getShape()[0]);

    Tensor C({n, p});

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {

            double sum = 0.0;

            for (int k = 0; k < m; k++) {
                sum += A.flat(k * n + i)
                     * B.flat(k * p + j);
            }

            C.flat(i * p + j) = sum;
        }
    }

    return C;
}

Tensor add(const Tensor& A, const Tensor& B) {

    return elementwise(A, B, [](double a, double b) {
        return a + b;
    });
}

Tensor multiply(const Tensor& A, const Tensor& B) {

    return elementwise(A, B, [](double a, double b) {
        return a * b;
    });
}

Tensor subtract(const Tensor& A, const Tensor& B) {

    return elementwise(A, B, [](double a, double b) {
        return a - b;
    });
}

Tensor scale(const Tensor& A, double scalar) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) * scalar;
    }

    return C;
}

Tensor sum(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total;

    return C;
}

Tensor mean(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total / A.size();

    return C;
}

Tensor transpose(const Tensor& A) {
    assert(A.ndim() == 2);

    int rows = A.getShape()[0];
    int cols = A.getShape()[1];

    Tensor result({cols, rows});

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result({j, i}) = A({i, j});
        }
    }

    return result;
}

