#include "tensor_ops.h"

#include <cassert>

Tensor matmul(const Tensor& A, const Tensor& B) {

    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.getShape()[1] == B.getShape()[0]);

    int m = A.getShape()[0];
    int n = A.getShape()[1];
    int p = B.getShape()[1];

    Tensor C({m, p});

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A({i, k}) * B({k, j});
            }

            C({i, j}) = sum;
        }
    }

    return C;
}

Tensor add(const Tensor& A, const Tensor& B) {

    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) + B.flat(i);
    }

    return C;
}

Tensor multiply(const Tensor& A, const Tensor& B) {

    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) * B.flat(i);
    }

    return C;
}

Tensor subtract(const Tensor& A, const Tensor& B) {
    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) - B.flat(i);
    }

    return C;
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