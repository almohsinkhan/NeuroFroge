#include "conv2d.h"
#include "cross_entropy.h"
#include "dataloader.h"
#include "dataset.h"
#include "flatten.h"
#include "linear.h"
#include "maxpool2d.h"
#include "relu.h"

#define private public
#include "sequential.h"
#undef private

#include "execution.h"
#include "softmax.h"
#include "trainer.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct LayerParameters {
    std::vector<double> weight;
    std::vector<double> bias;
    Conv2D* convolution = nullptr;
    Linear* linear = nullptr;
};

std::vector<LayerParameters> captureParameters(Sequential& model) {
    std::vector<LayerParameters> parameters;

    for (auto& layer : model.layers) {
        if (auto* convolution = dynamic_cast<Conv2D*>(layer.get())) {
            LayerParameters state;
            state.convolution = convolution;
            state.weight.resize(convolution->getWeight().size());
            state.bias.resize(convolution->getBias().size());

            for (int i = 0; i < convolution->getWeight().size(); i++) {
                state.weight[i] = convolution->getWeight().flat(i);
            }
            for (int i = 0; i < convolution->getBias().size(); i++) {
                state.bias[i] = convolution->getBias().flat(i);
            }
            parameters.push_back(state);
        } else if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
            LayerParameters state;
            state.linear = linear;
            state.weight.resize(linear->getWeight().size());
            state.bias.resize(linear->getBias().size());

            for (int i = 0; i < linear->getWeight().size(); i++) {
                state.weight[i] = linear->getWeight().flat(i);
            }
            for (int i = 0; i < linear->getBias().size(); i++) {
                state.bias[i] = linear->getBias().flat(i);
            }
            parameters.push_back(state);
        }
    }

    return parameters;
}

void restoreParameters(
    std::vector<LayerParameters>& parameters
) {
    for (LayerParameters& state : parameters) {
        Tensor* weight = nullptr;
        Tensor* bias = nullptr;

        if (state.convolution != nullptr) {
            weight = &state.convolution->getWeight();
            bias = &state.convolution->getBias();
        } else {
            weight = &state.linear->getWeight();
            bias = &state.linear->getBias();
        }

        for (int i = 0; i < weight->size(); i++) {
            weight->flat(i) = state.weight[i];
        }
        for (int i = 0; i < bias->size(); i++) {
            bias->flat(i) = state.bias[i];
        }
    }
}

Sequential createModel() {
    Sequential model;
    model.add<Conv2D>(1, 8, 3, 3);
    model.add<ReLU>();
    model.add<MaxPool2D>(2, 2, 2);
    model.add<Flatten>();
    model.add<Linear>(1352, 10);
    return model;
}

struct Metrics {
    double loss;
    double accuracy;
};

int argmax(
    const Tensor& probabilities,
    int sample,
    int batch_size
) {
    int best_class = 0;
    double best_value = probabilities.flat(sample);

    for (int class_index = 1;
         class_index < probabilities.getShape()[0];
         class_index++) {
        double value =
            probabilities.flat(class_index * batch_size + sample);

        if (value > best_value) {
            best_value = value;
            best_class = class_index;
        }
    }

    return best_class;
}

Metrics evaluate(
    Sequential& model,
    Dataset& dataset,
    CrossEntropyLoss& loss,
    int batch_size
) {
    DataLoader loader(
        dataset,
        batch_size,
        static_cast<int>(dataset.size()),
        false,
        BatchFormat::CNN
    );

    double total_loss = 0.0;
    int correct = 0;
    int samples = 0;

    while (loader.hasNext()) {
        Batch batch = loader.next();
        int current_batch_size = batch.images.getShape()[0];
        Tensor probabilities = softmax(model.forward(batch.images));
        Tensor targets({
            dataset.numClasses(),
            current_batch_size
        });

        for (int sample = 0;
             sample < current_batch_size;
             sample++) {
            targets.flat(
                batch.labels[sample] * current_batch_size + sample
            ) = 1.0;

            if (argmax(
                    probabilities,
                    sample,
                    current_batch_size
                ) == batch.labels[sample]) {
                correct++;
            }
        }

        total_loss +=
            loss.forward(probabilities, targets) *
            current_batch_size;
        samples += current_batch_size;
    }

    return {
        total_loss / samples,
        static_cast<double>(correct) / samples
    };
}

struct BenchmarkResult {
    int thread_count;
    std::size_t parallel_threshold;
    double total_seconds;
    double average_epoch_seconds;
    double average_batch_seconds;
    Metrics train;
    Metrics test;
};

BenchmarkResult runBenchmark(
    Sequential& model,
    Dataset& train_dataset,
    Dataset& test_dataset,
    int epochs,
    int batch_size,
    double learning_rate,
    int train_size,
    bool parallel,
    int parallel_threads
) {
    if (parallel) {
        execution::setThreadCount(parallel_threads);
        execution::setParallelThreshold(1);
    } else {
        execution::setThreadCount(1);
        execution::setParallelThreshold(
            std::numeric_limits<std::size_t>::max()
        );
    }

    CrossEntropyLoss loss;
    Trainer trainer(
        model,
        train_dataset,
        loss,
        BatchFormat::CNN
    );

    const int batch_count =
        (train_size + batch_size - 1) / batch_size;
    double total_seconds = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto start = std::chrono::steady_clock::now();
        trainer.fit(
            1,
            batch_size,
            learning_rate,
            train_size
        );
        auto finish = std::chrono::steady_clock::now();

        total_seconds += std::chrono::duration<double>(
            finish - start
        ).count();
    }

    Metrics train_metrics =
        evaluate(model, train_dataset, loss, batch_size);
    Metrics test_metrics =
        evaluate(model, test_dataset, loss, batch_size);

    return {
        execution::threadCount(),
        execution::parallelThreshold(),
        total_seconds,
        total_seconds / epochs,
        total_seconds / (epochs * batch_count),
        train_metrics,
        test_metrics
    };
}

void printResult(
    const std::string& name,
    const BenchmarkResult& result,
    bool parallel
) {
    std::cout
        << name << '\n'
        << "  execution: "
        << (parallel ? "OpenMP" : "serial") << '\n'
        << "  threads: " << result.thread_count << '\n'
        << "  parallel threshold: "
        << result.parallel_threshold << '\n'
        << "  total training time (s): "
        << result.total_seconds << '\n'
        << "  average epoch time (s): "
        << result.average_epoch_seconds << '\n'
        << "  average batch time (s): "
        << result.average_batch_seconds << '\n'
        << "  final training loss: "
        << result.train.loss << '\n'
        << "  final training accuracy: "
        << result.train.accuracy << '\n'
        << "  final test accuracy: "
        << result.test.accuracy << "\n\n";
}

}

int main() {
    const std::string data_root = "data/FashionMNIST/raw/";
    Dataset train_dataset = Dataset::fromIDX(
        data_root + "train-images-idx3-ubyte",
        data_root + "train-labels-idx1-ubyte"
    );
    Dataset test_dataset = Dataset::fromIDX(
        data_root + "t10k-images-idx3-ubyte",
        data_root + "t10k-labels-idx1-ubyte"
    );

    const int epochs = 1;
    const int batch_size = 64;
    const double learning_rate = 0.01;
    const int train_size = static_cast<int>(train_dataset.size());
    const int parallel_threads = 4;

    Sequential initial_model = createModel();
    std::vector<LayerParameters> initial_parameters =
        captureParameters(initial_model);

    Sequential serial_model = createModel();
    std::vector<LayerParameters> serial_parameters =
        captureParameters(serial_model);
    for (std::size_t i = 0; i < initial_parameters.size(); i++) {
        serial_parameters[i].weight = initial_parameters[i].weight;
        serial_parameters[i].bias = initial_parameters[i].bias;
    }
    restoreParameters(serial_parameters);

    Sequential parallel_model = createModel();
    std::vector<LayerParameters> parallel_parameters =
        captureParameters(parallel_model);
    for (std::size_t i = 0; i < initial_parameters.size(); i++) {
        parallel_parameters[i].weight = initial_parameters[i].weight;
        parallel_parameters[i].bias = initial_parameters[i].bias;
    }
    restoreParameters(parallel_parameters);

    CrossEntropyLoss warmup_loss;
    Trainer serial_warmup(
        serial_model,
        train_dataset,
        warmup_loss,
        BatchFormat::CNN
    );
    execution::setThreadCount(1);
    execution::setParallelThreshold(
        std::numeric_limits<std::size_t>::max()
    );
    serial_warmup.fit(
        1,
        batch_size,
        learning_rate,
        batch_size
    );
    restoreParameters(serial_parameters);

    Trainer parallel_warmup(
        parallel_model,
        train_dataset,
        warmup_loss,
        BatchFormat::CNN
    );
    execution::setThreadCount(parallel_threads);
    execution::setParallelThreshold(1);
    parallel_warmup.fit(
        1,
        batch_size,
        learning_rate,
        batch_size
    );
    restoreParameters(parallel_parameters);

    BenchmarkResult serial_result = runBenchmark(
        serial_model,
        train_dataset,
        test_dataset,
        epochs,
        batch_size,
        learning_rate,
        train_size,
        false,
        1
    );
    BenchmarkResult parallel_result = runBenchmark(
        parallel_model,
        train_dataset,
        test_dataset,
        epochs,
        batch_size,
        learning_rate,
        train_size,
        true,
        parallel_threads
    );

    std::cout << std::fixed << std::setprecision(6);
    printResult("Serial result", serial_result, false);
    printResult(
        "OpenMP result",
        parallel_result,
        true
    );
    std::cout
        << "Speedup (serial / OpenMP): "
        << serial_result.total_seconds /
           parallel_result.total_seconds
        << "x\n";
}
