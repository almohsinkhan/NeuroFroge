#include "tensor.h"

#include <iostream>
#include <cassert>

Tensor::Tensor(const std::vector<int>& shape)
    : shape(shape)
{
    strides.resize(shape.size());

    int stride = 1;

    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }

    data.resize(stride, 0.0);
}

int Tensor::ndim() const {
    return shape.size();
}

int Tensor::size() const {
    return data.size();
}

const std::vector<int>& Tensor::getShape() const {
    return shape;
}

double& Tensor::flat(int index) {
    assert(index >= 0 && index < data.size());
    return data[index];
}

double Tensor::flat(int index) const {
    assert(index >= 0 && index < data.size());
    return data[index];
}

double& Tensor::operator()(const std::vector<int>& indices) {

    assert(indices.size() == shape.size());

    int index = 0;

    for (int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0);
        assert(indices[i] < shape[i]);

        index += indices[i] * strides[i];
    }

    return data[index];
}

double Tensor::operator()(const std::vector<int>& indices) const {

    assert(indices.size() == shape.size());

    int index = 0;

    for (int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0);
        assert(indices[i] < shape[i]);

        index += indices[i] * strides[i];
    }

    return data[index];
}

void Tensor::printShape() const {

    std::cout << "(";

    for (int i = 0; i < shape.size(); i++) {
        std::cout << shape[i];

        if (i != shape.size() - 1)
            std::cout << ", ";
    }

    std::cout << ")\n";
}