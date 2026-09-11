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

    return 0;
}