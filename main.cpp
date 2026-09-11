#include <iostream>
#include <vector>
#include <cassert>

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


int main() {

    // 4-dimensional tensor
    Tensor x({2, 3, 4, 5});

    std::cout << "Shape: ";
    x.printShape();

    std::cout << "Dimensions: "
              << x.ndim() << "\n";

    std::cout << "Total elements: "
              << x.size() << "\n";


    // Access an element
    x({1, 2, 3, 4}) = 99;

    std::cout << "Value: "
              << x({1, 2, 3, 4})
              << "\n";


    // test matrix multiplication
    Tensor A({2, 3});
    Tensor B({3, 4});

    // Initialize A
    A({0, 0}) = 1; A({0, 1}) = 2; A({0, 2}) = 3;
    A({1, 0}) = 4; A({1, 1}) = 5; A({1, 2}) = 6;

    // Initialize B
    B({0, 0}) = 7;  B({0, 1}) = 8;  B({0, 2}) = 9;  B({0, 3}) = 10;
    B({1, 0}) = 11; B({1, 1}) = 12; B({1, 2}) = 13; B({1, 3}) = 14;
    B({2, 0}) = 15; B({2, 1}) = 16; B({2, 2}) = 17; B({2, 3}) = 18;

    Tensor C = matmul(A, B);    

    std::cout << "Result of A * B:\n";
    for (int i = 0; i < C.getShape()[0]; i++) {
        for (int j = 0; j < C.getShape()[1]; j++) {
            std::cout << C({i, j}) << " ";
        }
        std::cout << "\n";
    }


    return 0;
}