#include "tensor.h"

#include <iostream>
#include <cassert>
#include <algorithm>

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

bool Tensor::isBroadcastable(const Tensor& other) const {

    int ndimA = this->ndim();
    int ndimB = other.ndim();

    int maxDim = std::max(ndimA, ndimB);

    for (int i = 0; i < maxDim; i++) {

        // Get dimensions from the end, defaulting to 1 if the tensor has fewer dimensions
        int dimA = (i < ndimA)
            ? shape[ndimA - 1 - i]
            : 1;

        int dimB = (i < ndimB)
            ? other.shape[ndimB - 1 - i]
            : 1;

        // Check broadcasting rules
        if (dimA != dimB && dimA != 1 && dimB != 1)
            return false;
    }

    return true;
}

Tensor Tensor::broadcastTo(const std::vector<int>& newShape) const {

    // Check that broadcasting is possible
    Tensor target(newShape);

    int oldNdim = ndim();
    int newNdim = newShape.size();

    assert(newNdim >= oldNdim);

    // Check dimensions
    for (int i = 0; i < oldNdim; i++) {
        int oldDim = shape[oldNdim - 1 - i];
        int newDim = newShape[newNdim - 1 - i];

        assert(oldDim == newDim || oldDim == 1);
    }

    // Fill target tensor
    for (int i = 0; i < target.size(); i++) {

        int remaining = i;
        int oldIndex = 0;

        for (int d = newNdim - 1; d >= 0; d--) {

            int index = remaining % newShape[d];
            remaining /= newShape[d];

            // Corresponding old dimension
            int oldD = d - (newNdim - oldNdim);

            if (oldD >= 0) {

                // If old dimension is 1,
                // always use index 0.
                int oldIndexValue =
                    (shape[oldD] == 1) ? 0 : index;

                oldIndex += oldIndexValue * strides[oldD];
            }
        }

        target.flat(i) = data[oldIndex];
    }

    return target;
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