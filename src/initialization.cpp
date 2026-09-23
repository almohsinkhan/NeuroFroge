#include "initialization.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <cassert>



Tensor xavier_initialization(int in_features, int out_features) {
    Tensor weight({out_features, in_features});

    std::random_device rd;
    std::mt19937 gen(rd());

    double variance = 2.0 / (in_features + out_features);
    double stddev = std::sqrt(variance);

    std::normal_distribution<> d(0.0, stddev);

    for (int i = 0; i < weight.size(); i++) {
        weight.flat(i) = d(gen);
    }

    return weight;
}

// He Normal initialization
Tensor he_initialization(int in_features, int out_features) {
    Tensor weight({out_features, in_features});

    std::random_device rd;
    std::mt19937 gen(rd());

    double variance = 2.0 / in_features;
    double stddev = std::sqrt(variance);

    std::normal_distribution<> d(0.0, stddev);

    for (int i = 0; i < weight.size(); i++) {
        weight.flat(i) = d(gen);
    }

    return weight;
}

Tensor zeros(int out_features) {
    Tensor bias({out_features});

    for (int i = 0; i < bias.size(); i++) {
        bias.flat(i) = 0.0;
    }

    return bias;
}

// init weight with random normal distribution with mean 0 and stddev 0.1
Tensor random_normal_initialization(int in_features, int out_features) {
    Tensor weight({out_features, in_features});

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> d(0.0, 0.1);

    for (int i = 0; i < weight.size(); i++) {
        weight.flat(i) = d(gen);
    }

    return weight;
}