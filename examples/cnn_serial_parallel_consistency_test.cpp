#include "conv2d.h"
#include "cross_entropy.h"
#include "execution.h"
#include "flatten.h"
#include "linear.h"
#include "maxpool2d.h"
#include "relu.h"
#include "softmax.h"

#define private public
#include "sequential.h"
#undef private

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct LayerState {
    std::vector<double> weight;
    std::vector<double> bias;
};

struct StepResult {
    Tensor output;
    double loss;
    Tensor conv_grad_bias;
    Tensor conv_grad_weight;
    std::vector<LayerState> parameters;
};

struct Difference {
    double maximum_absolute;
    double maximum_relative;
};

Sequential createModel() {
    Sequential model;
    model.add<Conv2D>(1, 8, 3, 3);
    model.add<ReLU>();
    model.add<MaxPool2D>(2, 2, 2);
    model.add<Flatten>();
    model.add<Linear>(1352, 10);
    return model;
}

std::vector<LayerState> captureParameters(Sequential& model) {
    std::vector<LayerState> states;

    for (auto& layer : model.layers) {
        LayerState state;

        if (auto* convolution = dynamic_cast<Conv2D*>(layer.get())) {
            state.weight.resize(convolution->getWeight().size());
            state.bias.resize(convolution->getBias().size());

            for (int i = 0; i < convolution->getWeight().size(); i++) {
                state.weight[i] = convolution->getWeight().flat(i);
            }
            for (int i = 0; i < convolution->getBias().size(); i++) {
                state.bias[i] = convolution->getBias().flat(i);
            }
            states.push_back(state);
        } else if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
            state.weight.resize(linear->getWeight().size());
            state.bias.resize(linear->getBias().size());

            for (int i = 0; i < linear->getWeight().size(); i++) {
                state.weight[i] = linear->getWeight().flat(i);
            }
            for (int i = 0; i < linear->getBias().size(); i++) {
                state.bias[i] = linear->getBias().flat(i);
            }
            states.push_back(state);
        }
    }

    return states;
}

void restoreParameters(
    Sequential& model,
    const std::vector<LayerState>& states
) {
    int state_index = 0;

    for (auto& layer : model.layers) {
        Tensor* weight = nullptr;
        Tensor* bias = nullptr;

        if (auto* convolution = dynamic_cast<Conv2D*>(layer.get())) {
            weight = &convolution->getWeight();
            bias = &convolution->getBias();
        } else if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
            weight = &linear->getWeight();
            bias = &linear->getBias();
        } else {
            continue;
        }

        for (int i = 0; i < weight->size(); i++) {
            weight->flat(i) = states[state_index].weight[i];
        }
        for (int i = 0; i < bias->size(); i++) {
            bias->flat(i) = states[state_index].bias[i];
        }
        state_index++;
    }
}

Difference compare(
    const Tensor& left,
    const Tensor& right
) {
    assert(left.getShape() == right.getShape());
    Difference result{0.0, 0.0};

    for (int i = 0; i < left.size(); i++) {
        double absolute = std::abs(left.flat(i) - right.flat(i));
        double scale = std::max(
            std::abs(left.flat(i)),
            std::abs(right.flat(i))
        );
        double relative = scale == 0.0 ? absolute : absolute / scale;
        result.maximum_absolute =
            std::max(result.maximum_absolute, absolute);
        result.maximum_relative =
            std::max(result.maximum_relative, relative);
    }

    return result;
}

Difference compare(
    const std::vector<LayerState>& left,
    const std::vector<LayerState>& right
) {
    Difference result{0.0, 0.0};
    assert(left.size() == right.size());

    for (std::size_t layer = 0; layer < left.size(); layer++) {
        assert(left[layer].weight.size() == right[layer].weight.size());
        assert(left[layer].bias.size() == right[layer].bias.size());

        for (std::size_t i = 0; i < left[layer].weight.size(); i++) {
            double absolute = std::abs(
                left[layer].weight[i] - right[layer].weight[i]
            );
            double scale = std::max(
                std::abs(left[layer].weight[i]),
                std::abs(right[layer].weight[i])
            );
            double relative = scale == 0.0
                ? absolute
                : absolute / scale;
            result.maximum_absolute =
                std::max(result.maximum_absolute, absolute);
            result.maximum_relative =
                std::max(result.maximum_relative, relative);
        }

        for (std::size_t i = 0; i < left[layer].bias.size(); i++) {
            double absolute = std::abs(
                left[layer].bias[i] - right[layer].bias[i]
            );
            double scale = std::max(
                std::abs(left[layer].bias[i]),
                std::abs(right[layer].bias[i])
            );
            double relative = scale == 0.0
                ? absolute
                : absolute / scale;
            result.maximum_absolute =
                std::max(result.maximum_absolute, absolute);
            result.maximum_relative =
                std::max(result.maximum_relative, relative);
        }
    }

    return result;
}

void printDifference(
    const std::string& name,
    const Difference& difference
) {
    std::cout
        << "  " << name
        << ": max_abs=" << difference.maximum_absolute
        << ", max_rel=" << difference.maximum_relative
        << '\n';
}

Tensor makeInput(int batch_size) {
    Tensor input({batch_size, 1, 28, 28});

    for (int i = 0; i < input.size(); i++) {
        input.flat(i) =
            0.1 * std::sin(0.013 * static_cast<double>(i + 1));
    }

    return input;
}

Tensor makeTargets(int batch_size) {
    Tensor targets({10, batch_size});

    for (int sample = 0; sample < batch_size; sample++) {
        int label = (sample * 3 + 2) % 10;
        targets.flat(label * batch_size + sample) = 1.0;
    }

    return targets;
}

StepResult runStep(
    Sequential& model,
    const Tensor& input,
    const Tensor& targets,
    bool parallel
) {
    if (parallel) {
        execution::setThreadCount(4);
        execution::setParallelThreshold(1);
    } else {
        execution::setThreadCount(1);
        execution::setParallelThreshold(
            std::numeric_limits<std::size_t>::max()
        );
    }

    CrossEntropyLoss loss;
    model.zero_grad();
    Tensor output = model.forward(input);
    Tensor probabilities = softmax(output);
    double loss_value = loss.forward(probabilities, targets);
    Tensor dZ = loss.backward(probabilities, targets);
    model.backward(dZ);

    Conv2D* convolution =
        dynamic_cast<Conv2D*>(model.layers[0].get());

    StepResult result{
        output,
        loss_value,
        convolution->getGradBias(),
        convolution->getGradWeight(),
        {}
    };

    model.update(0.01, input.getShape()[0]);
    result.parameters = captureParameters(model);
    return result;
}

void assertWithinTolerance(
    const Difference& difference,
    double absolute_tolerance,
    double relative_tolerance
) {
    assert(
        difference.maximum_absolute <= absolute_tolerance ||
        difference.maximum_relative <= relative_tolerance
    );
}

void runCase(int batch_size) {
    Tensor input = makeInput(batch_size);
    Tensor targets = makeTargets(batch_size);

    Sequential initial_model = createModel();
    std::vector<LayerState> initial_parameters =
        captureParameters(initial_model);

    Sequential serial_model = createModel();
    Sequential parallel_model = createModel();
    restoreParameters(serial_model, initial_parameters);
    restoreParameters(parallel_model, initial_parameters);

    StepResult serial = runStep(
        serial_model,
        input,
        targets,
        false
    );
    StepResult parallel = runStep(
        parallel_model,
        input,
        targets,
        true
    );

    std::cout << "Batch size: " << batch_size << '\n';
    std::cout << std::setprecision(12);
    printDifference("forward output", compare(
        serial.output,
        parallel.output
    ));
    std::cout
        << "  loss: max_abs="
        << std::abs(serial.loss - parallel.loss)
        << ", max_rel="
        << std::abs(serial.loss - parallel.loss) /
           std::max(std::abs(serial.loss), std::abs(parallel.loss))
        << '\n';
    printDifference("Conv2D grad_bias", compare(
        serial.conv_grad_bias,
        parallel.conv_grad_bias
    ));
    printDifference("Conv2D grad_weight", compare(
        serial.conv_grad_weight,
        parallel.conv_grad_weight
    ));
    printDifference("updated parameters", compare(
        serial.parameters,
        parallel.parameters
    ));

    assertWithinTolerance(
        compare(serial.output, parallel.output),
        1e-12,
        1e-12
    );
    assert(std::abs(serial.loss - parallel.loss) < 1e-12);
    assertWithinTolerance(
        compare(serial.conv_grad_bias, parallel.conv_grad_bias),
        1e-12,
        1e-12
    );
    assertWithinTolerance(
        compare(serial.conv_grad_weight, parallel.conv_grad_weight),
        1e-12,
        1e-12
    );
    assertWithinTolerance(
        compare(serial.parameters, parallel.parameters),
        1e-12,
        1e-12
    );
}

void runConvDXCase(int batch_size) {
    Tensor input = makeInput(batch_size);
    Tensor dZ({batch_size, 8, 26, 26});

    for (int i = 0; i < dZ.size(); i++) {
        dZ.flat(i) =
            0.2 * std::cos(0.017 * static_cast<double>(i + 1));
    }

    Conv2D serial_conv(1, 8, 3, 3);
    Conv2D parallel_conv(1, 8, 3, 3);
    std::vector<LayerState> initial;
    LayerState state;
    state.weight.resize(serial_conv.getWeight().size());
    state.bias.resize(serial_conv.getBias().size());

    for (int i = 0; i < serial_conv.getWeight().size(); i++) {
        state.weight[i] = serial_conv.getWeight().flat(i);
    }
    for (int i = 0; i < serial_conv.getBias().size(); i++) {
        state.bias[i] = serial_conv.getBias().flat(i);
    }
    initial.push_back(state);

    for (int i = 0; i < parallel_conv.getWeight().size(); i++) {
        parallel_conv.getWeight().flat(i) = initial[0].weight[i];
    }
    for (int i = 0; i < parallel_conv.getBias().size(); i++) {
        parallel_conv.getBias().flat(i) = initial[0].bias[i];
    }

    execution::setThreadCount(1);
    execution::setParallelThreshold(
        std::numeric_limits<std::size_t>::max()
    );
    serial_conv.forward(input);
    Tensor serial_dX = serial_conv.backward(input, dZ);

    execution::setThreadCount(4);
    execution::setParallelThreshold(1);
    parallel_conv.forward(input);
    Tensor parallel_dX = parallel_conv.backward(input, dZ);

    Difference difference = compare(serial_dX, parallel_dX);
    printDifference("Conv2D dX", difference);
    assertWithinTolerance(difference, 1e-10, 1e-10);
}

}

int main() {
    std::cout
        << "OpenMP available: "
        << (execution::openmpAvailable() ? "yes" : "no")
        << '\n';

    runCase(2);
    runConvDXCase(2);
    runCase(8);
    runConvDXCase(8);
}
