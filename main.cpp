#include <iostream>
#include <vector>
#include <cassert>

class Tensor {
private:
    std::vector<int> shape;
    std::vector<double> data;

public:
    // Constructor
    Tensor(const std::vector<int>& shape)
        : shape(shape)
    {
        int size = 1;

        for (int dimension : shape) {
            size *= dimension;
        }

        data.resize(size, 0.0);
    }

    // Number of dimensions
    int ndim() const {
        return shape.size();
    }

    // Get shape
    const std::vector<int>& getShape() const {
        return shape;
    }

    // Total number of elements
    int size() const {
        return data.size();
    }

    // 1D indexing
    double& operator()(int i) {
        assert(ndim() == 1);

        return data[i];
    }

    // 2D indexing
    double& operator()(int i, int j) {
        assert(ndim() == 2);

        int cols = shape[1];

        return data[i * cols + j];
    }

    // 3D indexing
    double& operator()(int i, int j, int k) {
        assert(ndim() == 3);

        int dim1 = shape[1];
        int dim2 = shape[2];

        return data[i * dim1 * dim2
                  + j * dim2
                  + k];
    }

    // Print shape
    void printShape() const {
        std::cout << "(";

        for (int i = 0; i < shape.size(); i++) {
            std::cout << shape[i];

            if (i != shape.size() - 1)
                std::cout << ", ";
        }

        std::cout << ")\n";
    }

    // Print data
    void print() const {
        for (double value : data) {
            std::cout << value << " ";
        }

        std::cout << "\n";
    }
};


int main() {

    Tensor x({2, 3, 4});

    std::cout << "Dimensions: "
              << x.ndim() << "\n";

    std::cout << "Shape: ";
    x.printShape();

    std::cout << "Elements: "
              << x.size() << "\n";


    // Put some values inside

    x(0, 0, 0) = 10;
    x(0, 0, 1) = 20;
    x(1, 2, 3) = 99;


    std::cout << "\nValues:\n";

    std::cout << x(0, 0, 0) << "\n";
    std::cout << x(0, 0, 1) << "\n";
    std::cout << x(1, 2, 3) << "\n";
	
    return 0;
}