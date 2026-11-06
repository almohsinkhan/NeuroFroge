#include "conv2d.h"
#include "execution.h"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

Conv2D::Conv2D(
    int in_channels,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    const InitMethod init_method
)
    : weight({out_channels, in_channels, kernel_height, kernel_width}),
      bias({out_channels}),
      grad_weight({out_channels, in_channels, kernel_height, kernel_width}),
      grad_bias({out_channels}),
      in_channels(in_channels),
      out_channels(out_channels),
      kernel_height(kernel_height),
      kernel_width(kernel_width),
      stride(stride),
      padding(padding)
{
    int fan_in = in_channels * kernel_height * kernel_width;
    int fan_out = out_channels * kernel_height * kernel_width;

    if (init_method == InitMethod::XAVIER) {

        std::random_device rd;
        std::mt19937 gen(rd());
        double stddev = std::sqrt(
            2.0 / (fan_in + fan_out)
        );
        std::normal_distribution<> distribution(0.0, stddev);

        for (int i = 0; i < weight.size(); i++) {
            weight.flat(i) = distribution(gen);
        }

    } else if (init_method == InitMethod::HE) {

        weight = he_initialization(
            fan_in,
            out_channels
        ).reshape({
            out_channels,
            in_channels,
            kernel_height,
            kernel_width
        });

    } else if (init_method == InitMethod::RANDOM_NORMAL) {

        weight = random_normal_initialization(
            fan_in,
            out_channels
        ).reshape({
            out_channels,
            in_channels,
            kernel_height,
            kernel_width
        });

    } else {

        throw std::invalid_argument(
            "Invalid initialization method"
        );
    }

    bias = zeros(out_channels);
}

Tensor Conv2D::forward(const Tensor& input) {
    assert(input.ndim() == 4);

    const std::vector<int>& input_shape = input.getShape();
    assert(input_shape[1] == in_channels);
    assert(stride > 0);

    int batch_size = input_shape[0];
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    int output_height =
        (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width =
        (input_width + 2 * padding - kernel_width) / stride + 1;

    assert(output_height > 0);
    assert(output_width > 0);

    Tensor output({
        batch_size,
        out_channels,
        output_height,
        output_width
    });

    execution::parallelFor(
        static_cast<std::size_t>(batch_size) * out_channels,
        [&](std::size_t output_group) {
        int n = output_group / out_channels;
        int out_c = output_group % out_channels;

            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    double value = bias({out_c});

                    for (int in_c = 0; in_c < in_channels; in_c++) {
                        for (int kernel_h = 0;
                             kernel_h < kernel_height;
                             kernel_h++) {
                            for (int kernel_w = 0;
                                 kernel_w < kernel_width;
                                 kernel_w++) {
                                int input_h =
                                    out_h * stride + kernel_h - padding;
                                int input_w =
                                    out_w * stride + kernel_w - padding;

                                if (input_h >= 0 &&
                                    input_h < input_height &&
                                    input_w >= 0 &&
                                    input_w < input_width) {
                                    value +=
                                        input({
                                            n,
                                            in_c,
                                            input_h,
                                            input_w
                                        }) *
                                        weight({
                                            out_c,
                                            in_c,
                                            kernel_h,
                                            kernel_w
                                        });
                                }
                            }
                        }
                    }

                    output({
                        n,
                        out_c,
                        out_h,
                        out_w
                    }) = value;
                }
            }
        }
    );

    return output;
}

Tensor Conv2D::backward(
    const Tensor& input,
    const Tensor& dZ
) {
    assert(input.ndim() == 4);
    assert(dZ.ndim() == 4);

    const std::vector<int>& input_shape = input.getShape();
    const std::vector<int>& dZ_shape = dZ.getShape();

    assert(input_shape[1] == in_channels);
    assert(dZ_shape[0] == input_shape[0]);
    assert(dZ_shape[1] == out_channels);
    assert(stride > 0);

    int batch_size = input_shape[0];
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    int output_height =
        (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width =
        (input_width + 2 * padding - kernel_width) / stride + 1;

    assert(output_height > 0);
    assert(output_width > 0);
    assert(dZ_shape[2] == output_height);
    assert(dZ_shape[3] == output_width);

    Tensor dX(input_shape);

    execution::parallelFor(
        static_cast<std::size_t>(out_channels),
        [&](std::size_t out_channel_index) {
            int out_c = static_cast<int>(out_channel_index);
            double accumulated_gradient = 0.0;

            for (int n = 0; n < batch_size; n++) {
                for (int out_h = 0; out_h < output_height; out_h++) {
                    for (int out_w = 0; out_w < output_width; out_w++) {
                        accumulated_gradient +=
                            dZ({n, out_c, out_h, out_w});
                    }
                }
            }

            grad_bias({out_c}) += accumulated_gradient;
        }
    );

    execution::parallelFor(
        static_cast<std::size_t>(grad_weight.size()),
        [&](std::size_t weight_index) {
            int remaining = static_cast<int>(weight_index);
            int kernel_w = remaining % kernel_width;
            remaining /= kernel_width;
            int kernel_h = remaining % kernel_height;
            remaining /= kernel_height;
            int in_c = remaining % in_channels;
            int out_c = remaining / in_channels;

            double accumulated_gradient = 0.0;

            for (int n = 0; n < batch_size; n++) {
                for (int out_h = 0; out_h < output_height; out_h++) {
                    for (int out_w = 0; out_w < output_width; out_w++) {
                        int input_h =
                            out_h * stride + kernel_h - padding;
                        int input_w =
                            out_w * stride + kernel_w - padding;

                        if (input_h >= 0 &&
                            input_h < input_height &&
                            input_w >= 0 &&
                            input_w < input_width) {
                            accumulated_gradient +=
                                input({
                                    n,
                                    in_c,
                                    input_h,
                                    input_w
                                }) *
                                dZ({n, out_c, out_h, out_w});
                        }
                    }
                }
            }

            grad_weight({
                out_c,
                in_c,
                kernel_h,
                kernel_w
            }) += accumulated_gradient;
        }
    );

    const std::size_t output_work =
        static_cast<std::size_t>(batch_size) *
        out_channels *
        output_height *
        output_width;

#ifdef _OPENMP
    if (output_work >= execution::parallelThreshold()) {
        const int worker_capacity = omp_get_max_threads();
        std::vector<Tensor> worker_gradients;
        worker_gradients.reserve(worker_capacity);

        for (int worker = 0; worker < worker_capacity; worker++) {
            worker_gradients.emplace_back(input_shape);
        }

        execution::parallelFor(
            output_work,
            [&](std::size_t output_index) {
                int remaining = static_cast<int>(output_index);
                int out_w = remaining % output_width;
                remaining /= output_width;
                int out_h = remaining % output_height;
                remaining /= output_height;
                int out_c = remaining % out_channels;
                int n = remaining / out_channels;

                double upstream_gradient =
                    dZ({n, out_c, out_h, out_w});
                Tensor& worker_gradient =
                    worker_gradients[omp_get_thread_num()];

                for (int kernel_h = 0;
                     kernel_h < kernel_height;
                     kernel_h++) {
                    for (int kernel_w = 0;
                         kernel_w < kernel_width;
                         kernel_w++) {
                        int input_h =
                            out_h * stride + kernel_h - padding;
                        int input_w =
                            out_w * stride + kernel_w - padding;

                        if (input_h >= 0 &&
                            input_h < input_height &&
                            input_w >= 0 &&
                            input_w < input_width) {
                            for (int in_c = 0;
                                 in_c < in_channels;
                                 in_c++) {
                                worker_gradient({
                                    n,
                                    in_c,
                                    input_h,
                                    input_w
                                }) +=
                                    upstream_gradient *
                                    weight({
                                        out_c,
                                        in_c,
                                        kernel_h,
                                        kernel_w
                                    });
                            }
                        }
                    }
                }
            }
        );

        for (int worker = 0; worker < worker_capacity; worker++) {
            for (int i = 0; i < dX.size(); i++) {
                dX.flat(i) += worker_gradients[worker].flat(i);
            }
        }
    } else
#endif
    {
        for (int n = 0; n < batch_size; n++) {
            for (int out_c = 0; out_c < out_channels; out_c++) {
                for (int out_h = 0; out_h < output_height; out_h++) {
                    for (int out_w = 0; out_w < output_width; out_w++) {
                        double upstream_gradient =
                            dZ({n, out_c, out_h, out_w});

                        for (int kernel_h = 0;
                             kernel_h < kernel_height;
                             kernel_h++) {
                            for (int kernel_w = 0;
                                 kernel_w < kernel_width;
                                 kernel_w++) {
                                int input_h =
                                    out_h * stride + kernel_h - padding;
                                int input_w =
                                    out_w * stride + kernel_w - padding;

                                if (input_h >= 0 &&
                                    input_h < input_height &&
                                    input_w >= 0 &&
                                    input_w < input_width) {
                                    for (int in_c = 0;
                                         in_c < in_channels;
                                         in_c++) {
                                        dX({
                                            n,
                                            in_c,
                                            input_h,
                                            input_w
                                        }) +=
                                            upstream_gradient *
                                            weight({
                                                out_c,
                                                in_c,
                                                kernel_h,
                                                kernel_w
                                            });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dX;
}

void Conv2D::zero_grad() {
    for (int i = 0; i < grad_weight.size(); i++) {
        grad_weight.flat(i) = 0.0;
    }

    for (int i = 0; i < grad_bias.size(); i++) {
        grad_bias.flat(i) = 0.0;
    }
}

void Conv2D::update(
    double learning_rate,
    int batch_size
) {
    for (int i = 0; i < weight.size(); i++) {
        weight.flat(i) -=
            learning_rate *
            grad_weight.flat(i) /
            batch_size;
    }

    for (int i = 0; i < bias.size(); i++) {
        bias.flat(i) -=
            learning_rate *
            grad_bias.flat(i) /
            batch_size;
    }
}

Tensor& Conv2D::getWeight() {
    return weight;
}

Tensor& Conv2D::getBias() {
    return bias;
}

Tensor& Conv2D::getGradWeight() {
    return grad_weight;
}

Tensor& Conv2D::getGradBias() {
    return grad_bias;
}
