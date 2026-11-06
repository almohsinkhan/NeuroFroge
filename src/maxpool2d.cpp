#include "maxpool2d.h"
#include "execution.h"

#include <cassert>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

MaxPool2D::MaxPool2D(
    int kernel_height,
    int kernel_width,
    int stride
)
    : kernel_height(kernel_height),
      kernel_width(kernel_width),
      stride(stride)
{
    assert(kernel_height > 0);
    assert(kernel_width > 0);
    assert(stride > 0);
}

Tensor MaxPool2D::forward(const Tensor& input) {
    assert(input.ndim() == 4);

    const std::vector<int>& input_shape = input.getShape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    assert(input_height >= kernel_height);
    assert(input_width >= kernel_width);

    int output_height =
        (input_height - kernel_height) / stride + 1;
    int output_width =
        (input_width - kernel_width) / stride + 1;

    Tensor output({
        batch_size,
        channels,
        output_height,
        output_width
    });

    max_positions.resize(output.size());

    execution::parallelFor(
        static_cast<std::size_t>(output.size()),
        [&](std::size_t output_index) {
            int output_w = output_index % output_width;
            int remaining = output_index / output_width;
            int output_h = remaining % output_height;
            remaining /= output_height;
            int c = remaining % channels;
            int n = remaining / channels;

            int input_h_start = output_h * stride;
            int input_w_start = output_w * stride;
            int max_position = 0;
            double max_value = 0.0;
            bool has_value = false;

            for (int kernel_h = 0;
                 kernel_h < kernel_height;
                 kernel_h++) {
                for (int kernel_w = 0;
                     kernel_w < kernel_width;
                     kernel_w++) {
                    int input_h = input_h_start + kernel_h;
                    int input_w = input_w_start + kernel_w;
                    double value = input({
                        n,
                        c,
                        input_h,
                        input_w
                    });

                    if (!has_value || value > max_value) {
                        max_value = value;
                        max_position =
                            ((n * channels + c) * input_height +
                             input_h) * input_width +
                            input_w;
                        has_value = true;
                    }
                }
            }

            output.flat(output_index) = max_value;
            max_positions[output_index] = max_position;
        }
    );

    return output;
}

Tensor MaxPool2D::backward(
    const Tensor& input,
    const Tensor& dZ
) {
    assert(input.ndim() == 4);
    assert(dZ.ndim() == 4);

    const std::vector<int>& input_shape = input.getShape();
    const std::vector<int>& dZ_shape = dZ.getShape();

    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    assert(input_height >= kernel_height);
    assert(input_width >= kernel_width);

    int output_height =
        (input_height - kernel_height) / stride + 1;
    int output_width =
        (input_width - kernel_width) / stride + 1;

    assert(
        dZ_shape[0] == batch_size &&
        dZ_shape[1] == channels &&
        dZ_shape[2] == output_height &&
        dZ_shape[3] == output_width
    );

    if (max_positions.size() != dZ.size()) {
        throw std::logic_error(
            "MaxPool2D::backward() called before forward()"
        );
    }

    Tensor dX(input_shape);

#ifdef _OPENMP
    if (static_cast<std::size_t>(dZ.size()) >=
        execution::parallelThreshold()) {
        int worker_count = omp_get_max_threads();
        std::vector<std::vector<double>> worker_gradients(
            worker_count,
            std::vector<double>(dX.size(), 0.0)
        );

        execution::parallelFor(
            static_cast<std::size_t>(dZ.size()),
            [&](std::size_t index) {
                int worker = omp_get_thread_num();
                int position = max_positions[index];

                assert(position >= 0);
                assert(position < dX.size());
                worker_gradients[worker][position] += dZ.flat(index);
            }
        );

        for (int worker = 0; worker < worker_count; worker++) {
            for (int i = 0; i < dX.size(); i++) {
                dX.flat(i) += worker_gradients[worker][i];
            }
        }

        return dX;
    }
#endif

    for (int i = 0; i < dZ.size(); i++) {
        assert(max_positions[i] >= 0);
        assert(max_positions[i] < dX.size());
        dX.flat(max_positions[i]) += dZ.flat(i);
    }

    return dX;
}

void MaxPool2D::update(
    double,
    int
) {}

void MaxPool2D::zero_grad() {}
