#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <random>

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

    double& flat(int index) {
        assert(index >= 0 && index < data.size());
        return data[index];
    }

    double flat(int index) const {
        assert(index >= 0 && index < data.size());
        return data[index];
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


Tensor add(const Tensor& A, const Tensor& B) {

    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) + B.flat(i);
    }

    return C;
}


Tensor multiply(const Tensor& A, const Tensor& B) {

    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) * B.flat(i);
    }

    return C;
}


// subtract two tensors
Tensor subtract(const Tensor& A, const Tensor& B) {
    assert(A.getShape() == B.getShape());

    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) - B.flat(i);
    }

    return C;
}

// scale a tensor by a scalar
Tensor scale(const Tensor& A, double scalar) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) * scalar;
    }

    return C;
}

// sum all elements of a tensor
Tensor sum(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total;

    return C;
}

// mean elements of a tensor
Tensor mean(const Tensor& A) {
    double total = 0.0;

    for (int i = 0; i < A.size(); i++) {
        total += A.flat(i);
    }

    Tensor C({1});
    C.flat(0) = total / A.size();

    return C;
}

// implement activation functions: relu, sigmoid, tanh
Tensor relu(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = std::max(0.0, A.flat(i));
    }

    return C;
}

Tensor sigmoid(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = 1.0 / (1.0 + std::exp(-A.flat(i)));
    }

    return C;
}


Tensor tanh(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = std::tanh(A.flat(i));
    }

    return C;
}

Tensor transpose(const Tensor& A) {
    assert(A.ndim() == 2);

    int rows = A.getShape()[0];
    int cols = A.getShape()[1];

    Tensor result({cols, rows});

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result({j, i}) = A({i, j});
        }
    }

    return result;
}

// wait initialize the weights and biases 
// Xavier/Glorot Initialization 
/*initialization weight by drawing from a normal distribution 
with mean 0 and specific variance*/

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

enum class InitMethod {
    RANDOM_NORMAL,
    XAVIER,
    HE
};

class Linear {
    private:
        Tensor weight;
        Tensor bias;
        
        Tensor grad_weight;
        Tensor grad_bias;

    public:
        Linear(int in_features, int out_features, const InitMethod init_method = InitMethod::RANDOM_NORMAL) 
            :weight({out_features, in_features}),
            bias({out_features}),
            grad_weight({out_features, in_features}),
            grad_bias({out_features})
        {

            if (init_method == InitMethod::XAVIER) {
                weight = xavier_initialization(in_features, out_features);
            } else if (init_method == InitMethod::HE) {
                weight = he_initialization(in_features, out_features);
            } else if (init_method == InitMethod::RANDOM_NORMAL) {
                weight = random_normal_initialization(in_features, out_features);
            } else {
                throw std::invalid_argument("Invalid initialization method");
            }

            bias = zeros(out_features);
        }

        Tensor forward(const Tensor& input) {
            assert(input.ndim() == 2);
            assert(input.getShape()[0] == weight.getShape()[1]);

            // multiply input with weight
            Tensor output = matmul(weight, input);

            // add bias
            for (int i = 0; i < output.size(); i++) {
                output({i, 0}) += bias({i});
            }
            return output;
        }

        Tensor backward(const Tensor& input, const Tensor& dZ) {
            // dW = dZ × X^T
            Tensor input_T = transpose(input);
            grad_weight = matmul(dZ, input_T);

            // db = dZ
            for (int i = 0; i < bias.size(); i++) {
                grad_bias.flat(i) = dZ.flat(i);
            }

            // dX = W^T × dZ
            Tensor weight_T = transpose(weight);
            Tensor dX = matmul(weight_T, dZ);

            return dX;
       }

       void update(double learning_rate) {
            // W = W - learning_rate * dW
            for (int i = 0; i < weight.size(); i++) {
                weight.flat(i) -= learning_rate * grad_weight.flat(i);
            }

            // b = b - learning_rate * db
            for (int i = 0; i < bias.size(); i++) {
                bias.flat(i) -= learning_rate * grad_bias.flat(i);
            }
        }

        Tensor& getWeight() {
            return weight;
        }

        Tensor& getBias() {
            return bias;
        }

        Tensor& getGradWeight() {
            return grad_weight;
        }

        Tensor& getGradBias() {
            return grad_bias;
        }
};

// binary cross entropy loss function
double binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    // ensure predictions and targets have the same shape
    assert(predictions.getShape() == targets.getShape());

    // initialize loss to 0
    double loss = 0.0;

    for (int i = 0; i < predictions.size(); i++) {
        double p = predictions.flat(i);
        double y = targets.flat(i);

        // ensure predictions are in the range (0, 1)
        assert(p > 0.0 && p < 1.0);

        // compute binary cross entropy loss
        loss += - (y * std::log(p) + (1 - y) * std::log(1 - p));
    }
    return loss / predictions.size();
}

// relu derivative
Tensor relu_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        C.flat(i) = A.flat(i) > 0.0 ? 1.0 : 0.0;
    }

    return C;
}

// sigmoid derivative
Tensor sigmoid_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        double s = 1.0 / (1.0 + std::exp(-A.flat(i)));
        C.flat(i) = s * (1 - s);
    }

    return C;
}

// tanh derivative
Tensor tanh_derivative(const Tensor& A) {
    Tensor C(A.getShape());

    for (int i = 0; i < A.size(); i++) {
        double t = std::tanh(A.flat(i));
        C.flat(i) = 1 - t * t;
    }

    return C;
}

//binary cross entropy backward function
Tensor binary_cross_entropy_backward(const Tensor& predictions, const Tensor& targets) {
    assert(predictions.getShape() == targets.getShape());

    Tensor grad(predictions.getShape());

    // Compute the gradient of the binary cross entropy loss with respect to the predictions
    for (int i = 0; i < predictions.size(); i++){
        grad.flat(i) = predictions.flat(i) - targets.flat(i);
    }

    return grad;
}


int main() {

    // create linear layers 
    Linear layer1(2, 3, InitMethod::HE);
    Linear layer2(3, 1, InitMethod::HE);

    // XOR inputs and targets 
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4] = {
        0, 1, 1, 0
    };

    // training parameters
    double learning_rate = 0.1;
    int epochs = 10000;

    // training loop
    for (int epoch = 0; epoch < epochs; epoch++) {

        double total_loss = 0.0;

        // loop over each sample in the dataset
        for (int sample = 0; sample < 4; sample++) {

            Tensor input({2, 1});
            input({0, 0}) = inputs[sample][0];
            input({1, 0}) = inputs[sample][1];

            Tensor target({1, 1});
            target({0, 0}) = targets[sample];

            Tensor z1 = layer1.forward(input);
            Tensor hidden = relu(z1);
            Tensor z2 = layer2.forward(hidden);
            Tensor prediction = sigmoid(z2);

            double loss = binary_cross_entropy(prediction, target);
            total_loss += loss;

            Tensor dZ2 =
                binary_cross_entropy_backward(prediction, target);

            Tensor dHidden =
                layer2.backward(hidden, dZ2);

            Tensor relu_grad =
                relu_derivative(z1);

            Tensor dZ1 =
                multiply(dHidden, relu_grad);

            Tensor dInput =
                layer1.backward(input, dZ1);

            layer1.update(learning_rate);
            layer2.update(learning_rate);
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch
                      << " Loss: "
                      << total_loss / 4
                      << "\n";
        }
    }

    std::cout << "\nXOR Predictions\n";

    for (int sample = 0; sample < 4; sample++) {

        Tensor input({2, 1});
        input({0, 0}) = inputs[sample][0];
        input({1, 0}) = inputs[sample][1];

        Tensor z1 = layer1.forward(input);
        Tensor hidden = relu(z1);
        Tensor z2 = layer2.forward(hidden);
        Tensor prediction = sigmoid(z2);

        std::cout << inputs[sample][0]
                  << " XOR "
                  << inputs[sample][1]
                  << " = "
                  << prediction.flat(0)
                  << "\n";
    }

    return 0;
}