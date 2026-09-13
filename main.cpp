#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

class Tensor {
private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<double> data;

public:
    Tensor(const std::vector<int>& shape)
        : shape(shape)
    {
        // Calculate strides
        strides.resize(shape.size());

        int stride = 1;

        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        // Allocate memory
        data.resize(stride, 0.0);
    }

    int ndim() const {
        return shape.size();
    }

    int size() const {
        return data.size();
    }

    const std::vector<int>& getShape() const {
        return shape;
    }

    double& flat(int index) {
        assert(index >= 0 && index < data.size());
        return data[index];
    }

    double flat(int index) const {
        assert(index >= 0 && index < data.size());
        return data[index];
    }

    // Generic N-dimensional indexing
    double& operator()(const std::vector<int>& indices) {

        assert(indices.size() == shape.size());

        int index = 0;

        for (int i = 0; i < shape.size(); i++) {

            assert(indices[i] >= 0);
            assert(indices[i] < shape[i]);

            index += indices[i] * strides[i];
        }

        return data[index];
    }

    double operator()(const std::vector<int>& indices) const {
    assert(indices.size() == shape.size());

    int index = 0;

    for (int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0);
        assert(indices[i] < shape[i]);

        index += indices[i] * strides[i];
    }

    return data[index];
    }

    void printShape() const {

        std::cout << "(";

        for (int i = 0; i < shape.size(); i++) {

            std::cout << shape[i];

            if (i != shape.size() - 1)
                std::cout << ", ";
        }

        std::cout << ")\n";
    }
};

// for now we are only implementing 2D matrix multiplication
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


// subtract two tensors
Tensor subtract(const Tensor& A, const Tensor& B) {
    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) - B.flat(i);
    }

    return C;
}

// scale a tensor by a scalar
Tensor scale(const Tensor& A, double scalar) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) * scalar;
    }

    return C;
}

// sum all elements of a tensor
Tensor sum(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total;

    return C;
}

// mean elements of a tensor
Tensor mean(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total / A.size();

    return C;
}

// implement activation functions: relu, sigmoid, tanh
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

// create a custom activation function: tanh

Tensor tanh(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = std::tanh(A.flat(i));
    }

    return C;
}


int main() {

    Tensor A({2, 3});

    A({0, 0}) = -2;
    A({0, 1}) = 3;
    A({0, 2}) = -1;

    A({1, 0}) = 5;
    A({1, 1}) = -4;
    A({1, 2}) = 2;

    Tensor B = tanh(A);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << B({i, j}) << " ";
        }
        std::cout << "\n";
    }
}