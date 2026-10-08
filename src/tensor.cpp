#include "tensor.h"

#include <iostream>
#include <cassert>
#include <algorithm>

Tensor::Tensor(
    const std::vector<int>& shape,
    const std::vector<int>& strides,
    std::shared_ptr<std::vector<double>> data
)
    : shape(shape),
      strides(strides),
      data(data)
{
}

Tensor::Tensor(const std::vector<int>& shape)
    : shape(shape)
{
    strides.resize(shape.size());

    int stride = 1;

    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }

    data = std::make_shared<std::vector<double>>(stride, 0.0);
}

int Tensor::ndim() const {
    return shape.size();
}

int Tensor::size() const {

    int total = 1;

    for (int dim : shape)
        total *= dim;

    return total;
}

const std::vector<int>& Tensor::getShape() const {
    return shape;
}

double& Tensor::flat(int index) {

    assert(index >= 0 && index < size());

    // Fast path for contiguous tensors
    if (isContiguous())
        return (*data)[index];

    // View path
    int remaining = index;
    int physicalIndex = 0;

    for (int i = ndim() - 1; i >= 0; i--) {

        int coordinate = remaining % shape[i];
        remaining /= shape[i];

        physicalIndex += coordinate * strides[i];
    }

    assert(physicalIndex >= 0 &&
           physicalIndex < data->size());

    return (*data)[physicalIndex];
}

double Tensor::flat(int index) const {

    assert(index >= 0 && index < size());

    if (isContiguous())
        return (*data)[index];

    int remaining = index;
    int physicalIndex = 0;

    for (int i = ndim() - 1; i >= 0; i--) {

        int coordinate = remaining % shape[i];
        remaining /= shape[i];

        physicalIndex += coordinate * strides[i];
    }

    assert(physicalIndex >= 0 &&
           physicalIndex < data->size());

    return (*data)[physicalIndex];
}

double& Tensor::operator()(const std::vector<int>& indices) {

    assert(indices.size() == shape.size());

    int index = 0;

    for (int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0);
        assert(indices[i] < shape[i]);

        index += indices[i] * strides[i];
    }

    return (*data)[index];
}

double Tensor::operator()(const std::vector<int>& indices) const {

    assert(indices.size() == shape.size());

    int index = 0;

    for (int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0);
        assert(indices[i] < shape[i]);

        index += indices[i] * strides[i];
    }

    return (*data)[index];
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

    int oldNdim = ndim();
    int newNdim = newShape.size();

    assert(newNdim >= oldNdim);

    std::vector<int> newStrides(newNdim);

    for (int i = 0; i < newNdim; i++) {

        int newD = newNdim - 1 - i;

        if (i < oldNdim) {

            int oldD = oldNdim - 1 - i;

            int oldDim = shape[oldD];
            int newDim = newShape[newD];

            assert(oldDim == newDim || oldDim == 1);

            if (oldDim == 1 && newDim > 1)
                newStrides[newD] = 0;
            else
                newStrides[newD] = strides[oldD];

        } else {

            // New dimension added to the left
            newStrides[newD] = 0;
        }
    }

    return Tensor(newShape, newStrides, data);
}

bool Tensor::isContiguous() const {

    int expectedStride = 1;

    for (int i = ndim() - 1; i >= 0; i--) {

        if (strides[i] != expectedStride)
            return false;

        expectedStride *= shape[i];
    }

    return true;
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

Tensor Tensor::reshape(const std::vector<int>& newShape) const {

    int totalSize = size();

    int knownSize = 1;
    int unknownIndex = -1;

    for (int i = 0; i < newShape.size(); i++) {

        if (newShape[i] == -1) {

            assert(unknownIndex == -1);
            unknownIndex = i;

        } else {

            assert(newShape[i] > 0);
            knownSize *= newShape[i];
        }
    }

    std::vector<int> finalShape = newShape;

    if (unknownIndex != -1) {

        assert(totalSize % knownSize == 0);

        finalShape[unknownIndex] = totalSize / knownSize;
    }

    int newSize = 1;

    for (int dim : finalShape)
        newSize *= dim;

    assert(newSize == totalSize);
    assert(isContiguous());

    std::vector<int> newStrides(finalShape.size());

    int stride = 1;

    for (int i = finalShape.size() - 1; i >= 0; i--) {

        newStrides[i] = stride;
        stride *= finalShape[i];
    }

    return Tensor(finalShape, newStrides, data);
}

const std::vector<int>& Tensor::getStrides() const {
    return strides;
}

std::shared_ptr<std::vector<double>> Tensor::getData() const {
    return data;
}