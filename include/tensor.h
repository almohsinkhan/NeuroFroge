#pragma once

#include <vector>
#include <memory>

class Tensor {
private:
    std::vector<int> shape;
    std::vector<int> strides;
    bool contiguous;
    std::shared_ptr<std::vector<double>> data;

public:
    Tensor(const std::vector<int>& shape);

    // Constructor for creating a tensor with a specific data pointer
    Tensor(
        const std::vector<int>& shape,
        const std::vector<int>& strides,
        std::shared_ptr<std::vector<double>> data
    );

    Tensor reshape(const std::vector<int>& newShape) const;

    int ndim() const;
    int size() const;

    const std::vector<int>& getShape() const;

    double& flat(int index);
    double flat(int index) const;

    double& operator()(const std::vector<int>& indices);
    double operator()(const std::vector<int>& indices) const;

    bool isBroadcastable(const Tensor& other) const;
    Tensor broadcastTo(const std::vector<int>& newShape) const;

    bool isContiguous() const;

    const std::vector<int>& getStrides() const;

    std::shared_ptr<std::vector<double>> getData() const;

    double* rowData();

    const double* rowData() const;
    void printShape() const;
};