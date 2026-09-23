#pragma once

#include <vector>

class Tensor {
private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<double> data;

public:
    Tensor(const std::vector<int>& shape);

    int ndim() const;
    int size() const;

    const std::vector<int>& getShape() const;

    double& flat(int index);
    double flat(int index) const;

    double& operator()(const std::vector<int>& indices);
    double operator()(const std::vector<int>& indices) const;

    void printShape() const;
};