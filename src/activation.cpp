#include "activation.h"
#include <iostream>
#include <cmath>
#include <cassert>


Tensor relu(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = std::max(0.0, A.flat(i));
    }

    return C;
}

Tensor sigmoid(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = 1.0 / (1.0 + std::exp(-A.flat(i)));
    }

    return C;
}

Tensor tanh(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = std::tanh(A.flat(i));
    }

    return C;
}


// relu derivative
Tensor relu_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) > 0.0 ? 1.0 : 0.0;
    }

    return C;
}

// sigmoid derivative
Tensor sigmoid_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        double s = 1.0 / (1.0 + std::exp(-A.flat(i)));
        C.flat(i) = s * (1 - s);
    }

    return C;
}

// tanh derivative
Tensor tanh_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        double t = std::tanh(A.flat(i));
        C.flat(i) = 1 - t * t;
    }

    return C;
}
