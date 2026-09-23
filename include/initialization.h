#pragma once

#include "tensor.h"

enum class InitMethod {
    RANDOM_NORMAL,
    XAVIER,
    HE
};

Tensor xavier_initialization(int in_features, int out_features);

Tensor he_initialization(int in_features, int out_features);

Tensor random_normal_initialization(int in_features, int out_features);

Tensor zeros(int out_features);