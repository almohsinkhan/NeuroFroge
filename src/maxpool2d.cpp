#include "maxpool2d.h"

#include <cassert>
#include <stdexcept>

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

    int output_index = 0;

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            for (int output_h = 0;
                 output_h < output_height;
                 output_h++) {
                for (int output_w = 0;
                     output_w < output_width;
                     output_w++) {
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
                    output_index++;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(
    const Tensor&,
    const Tensor&
) {
    throw std::logic_error(
        "MaxPool2D::backward() is not implemented"
    );
}

void MaxPool2D::update(
    double,
    int
) {}

void MaxPool2D::zero_grad() {}
