# NeuroFroge

NeuroFroge is a neural-network framework implemented from scratch in C++.
The project is intended to make the data representation, forward pass,
backward pass, parameter update, and CPU execution strategy explicit.

The project is not intended to compete with production frameworks such as
PyTorch. Its purpose is to understand the decisions that those frameworks
hide behind high-level APIs. The implementation is being developed in stages:
first tensor and optimization fundamentals, then convolutional networks,
then broader CPU execution improvements, and eventually Transformer
components.

## Project status

The project currently supports:

- Stride-aware N-dimensional tensors.
- Shared-storage tensor views, broadcasting, and reshape operations.
- Matrix multiplication and transpose-aware matrix multiplication.
- Linear, Conv2D, ReLU, Sigmoid, Tanh, MaxPool2D, and Flatten layers.
- Sequential models with forward, backward, gradient reset, and update steps.
- Cross-entropy loss and softmax classification.
- Mini-batch training for both MLP and CNN data layouts.
- CSV and raw IDX dataset loading.
- Fashion-MNIST CNN training.
- Optional OpenMP execution with a serial fallback.
- Numerical gradient checks and serial-versus-parallel validation tests.

The current implementation is a learning framework. It has no formal build
system, automatic dataset download, GPU backend, or production checkpointing
system. Examples are compiled directly with `g++`.

## Design goals

### Learn the complete execution path

Every major operation is implemented in repository code rather than delegated
to an external machine-learning library. This makes it possible to inspect
how tensor indices map to memory, how gradients are accumulated, and when
parameters are updated.

### Preserve a simple reference implementation

Parallel implementations are added beside, or around, serial algorithms
rather than replacing the reference behavior without a comparison path. The
serial path is important because it provides a clear correctness baseline for
tests and numerical debugging.

### Prefer explicit ownership over synchronization

When multiple workers can contribute to one result, the implementation uses
local accumulators or private buffers. This avoids atomics in the main
numerical paths and makes write ownership clear. The tradeoff is additional
temporary memory for operations such as MaxPool2D and Conv2D backward.

### Keep data layout contracts explicit

MLP and CNN layers do not consume the same batch layout. The framework
therefore represents the distinction with `BatchFormat` instead of silently
guessing from tensor shapes. This preserves existing MLP behavior while
allowing CNN batches to use the conventional NCHW layout.

## Architecture

### Tensor representation

`Tensor` stores:

- A shape vector.
- A stride vector.
- Shared backing storage.
- A contiguous flag derived from the shape and strides.

Contiguous tensors use direct flat indexing. Views use their strides when
accessed through `operator()` or `flat()`. This allows operations to read
non-contiguous and broadcast views without copying them.

`rowData()` is restricted to contiguous tensors because a row pointer cannot
represent arbitrary strides safely. Operations that support views use
stride-aware access instead.

This design was chosen to expose the relationship between logical tensor
coordinates and physical storage. It also lets tests verify that layers do not
accidentally depend on contiguous memory.

### Module and Sequential

All trainable and non-trainable layers implement the `Module` interface:

- `forward()`
- `backward()`
- `zero_grad()`
- `update()`
- `name()`

`Sequential` stores modules in execution order. During the forward pass it
caches each layer input. During the backward pass it visits layers in reverse
order and passes each returned gradient to the preceding layer.

Caching inputs in `Sequential` keeps the layer interface small. Each layer can
calculate its local gradient without exposing its internal activation cache to
the rest of the framework.

### Training loop

`Trainer` performs the following operations for every batch:

1. Reset accumulated gradients.
2. Create one-hot targets.
3. Run the model forward pass.
4. Apply softmax.
5. Calculate loss and batch accuracy.
6. Calculate the loss gradient.
7. Run the model backward pass.
8. Update parameters using the accumulated gradients and batch size.

The update occurs after backward propagation for each batch. Gradients are
reset before the next batch, while each layer still supports accumulation
across multiple backward calls before an update.

The trainer reports every five batches and at the end of each epoch. MLP
batches use `[features, batch]`; CNN batches use `[batch, channels, height,
width]`. The trainer selects the batch dimension from `BatchFormat` so the
original MLP behavior remains unchanged.

## Implemented layers

### Linear

`Linear` implements a fully connected layer with weights, bias, accumulated
gradients, initialization, backward propagation, and SGD updates. It is the
basic trainable layer used by the original MLP workflow and by the final
classifier in the CNN example.

### Conv2D

`Conv2D` uses the NCHW convention:

```text
input:  [batch, input_channels, height, width]
weight: [output_channels, input_channels, kernel_height, kernel_width]
output: [batch, output_channels, output_height, output_width]
```

The output dimensions are calculated as:

```text
output_height =
    (input_height + 2 * padding - kernel_height) / stride + 1

output_width =
    (input_width + 2 * padding - kernel_width) / stride + 1
```

The forward pass supports stride and padding. The backward pass calculates
`grad_bias`, `grad_weight`, and `dX`, and preserves accumulated gradients
until `zero_grad()` is called.

Initialization supports Xavier, He, and random-normal methods. Xavier and He
use convolution-specific fan-in and fan-out values that include the input or
output channel count and kernel area.

Conv2D backward parallelization was introduced in stages:

1. `grad_bias` is parallelized over output channels. Each worker owns one
   complete bias element and reduces its batch and spatial contributions
   locally.
2. `grad_weight` is parallelized over complete weight elements. Each worker
   owns one `(output_channel, input_channel, kernel_height, kernel_width)`
   element and uses a local accumulator.
3. `dX` uses private per-worker tensors. Workers retain the existing
   output-oriented scatter calculation, but write only to their private
   tensor. The private tensors are reduced into the final result after the
   parallel loop.

The third decision is necessary because overlapping convolution windows can
write to the same input-gradient element. Direct concurrent writes would
create races. The private-buffer approach avoids atomics and preserves the
serial scatter path for small workloads and non-OpenMP builds.

### ReLU

ReLU applies an element-wise maximum with zero. Forward and backward operate
on logical tensor elements through stride-aware access. Independent elements
are processed with the shared execution utility when the tensor is large
enough.

### MaxPool2D

MaxPool2D uses NCHW input and caches the flattened input position of the first
maximum in every pooling window. The first-maximum rule makes ties
deterministic.

Forward output elements are independent and are parallelized directly.
Backward uses private input-gradient buffers because overlapping windows can
contribute to the same input element. The buffers are reduced after the
parallel loop.

### Flatten

Flatten converts NCHW data to the layout expected by `Linear`:

```text
[batch, channels, height, width]
    -> [channels * height * width, batch]
```

This is a logical reordering, not merely a reshape. The original input shape
is cached so backward propagation can restore the NCHW shape. The
implementation uses logical indexing so non-contiguous views remain
supported.

### Activations and loss

The project includes ReLU, Sigmoid, Tanh, and Softmax. Cross-entropy loss
provides both the scalar loss and its gradient for classification.

## Shared execution utility

`execution::parallelFor()` is the common entry point for optional CPU
parallelism. It provides:

- OpenMP detection at compile time.
- A configurable work threshold.
- Runtime thread-count configuration.
- A serial fallback when OpenMP is unavailable.
- A serial path for work below the configured threshold.

The utility was introduced to avoid repeating OpenMP configuration and
threshold logic in every layer. Tensor operations and layers can therefore
focus on their ownership rules while the execution policy stays centralized.

The current parallelized operations are:

- Conv2D forward.
- Conv2D backward gradient calculations.
- ReLU forward and backward.
- Flatten forward and backward.
- MaxPool2D forward and backward.
- Matrix multiplication and its transpose-aware variants.

The inner reduction loops remain sequential where that preserves ownership and
keeps the implementation straightforward. For example, matrix multiplication
parallelizes output elements but keeps each output element's `k` reduction
local and sequential.

## Matrix multiplication

The project supports:

- `matmul()`
- `matmul_transpose_right()`
- `matmul_transpose_left()`

Each operation computes independent output elements in parallel when the work
threshold is reached. The inner reduction is kept sequential, and each output
is written once from a local accumulator. Access uses stride-aware paths for
non-contiguous views. `rowData()` is used only after contiguity has been
validated.

An existing benchmark measured an improvement from approximately 42.92
seconds in the serial path to 12.92 seconds with OpenMP for its workload.
This result is workload- and machine-dependent and is included as an
engineering observation rather than a universal performance claim.

## Data loading

`Dataset` supports:

- CSV image and label data.
- Raw IDX image and label files.

The IDX reader validates magic numbers, dimensions, counts, label ranges,
truncated data, and trailing data. IDX image bytes are stored compactly and
converted to normalized floating-point tensors when samples are requested.

The current reader expects decompressed IDX files. The repository does not
download or decompress datasets automatically.

`DataLoader` supports:

```text
BatchFormat::MLP
    images: [features, batch]

BatchFormat::CNN
    images: [batch, channels, height, width]
```

The default remains `BatchFormat::MLP` for compatibility with the original
training examples. CNN mode accepts 28 by 28 samples and produces NCHW
batches for Conv2D.

DataLoader shuffles sample indices by default and keeps each sample paired with
its label. The existing constructor uses a nondeterministic seed. For
reproducible shuffling, use the overload with an explicit seed:

```cpp
DataLoader loader(
    dataset,
    batch_size,
    max_samples,
    true,
    BatchFormat::MLP,
    12345
);
```

Calling `reset()` starts the next epoch and advances the same generator, so
identically configured loaders produce the same sequence of shuffled epochs.
Set `shuffle` to `false` to preserve dataset order.

All augmentation settings can also be grouped in an
`AugmentationConfig`. Its defaults keep flipping, cropping, brightness,
contrast, and normalization disabled while retaining nondeterministic seeding:

```cpp
AugmentationConfig augmentation;
augmentation.seed = 12345;
augmentation.horizontal_flip_probability = 0.5;
augmentation.crop_height = 24;
augmentation.crop_width = 24;
augmentation.brightness_probability = 0.5;
augmentation.brightness_min_factor = 0.8;
augmentation.brightness_max_factor = 1.2;
augmentation.contrast_probability = 0.5;
augmentation.contrast_min_factor = 0.8;
augmentation.contrast_max_factor = 1.2;
augmentation.rotation_probability = 0.5;
augmentation.rotation_min_degrees = -15.0;
augmentation.rotation_max_degrees = 15.0;
augmentation.vertical_flip_probability = 0.5;
augmentation.translation_min_horizontal = -2;
augmentation.translation_max_horizontal = 2;
augmentation.translation_min_vertical = -2;
augmentation.translation_max_vertical = 2;
augmentation.resize_height = 28;
augmentation.resize_width = 28;
augmentation.noise_mean = 0.0;
augmentation.noise_stddev = 0.01;
augmentation.blur_kernel_size = 3;
augmentation.blur_sigma = 1.0;

DataLoader loader(
    dataset,
    batch_size,
    max_samples,
    true,
    BatchFormat::CNN,
    augmentation
);
```

Set `channel_means` and `channel_stddevs` on the config to enable
per-channel normalization. The config constructor uses the same augmentation
ordering and seeded generator as the positional constructor.

The spatial pipeline is ordered as crop, resize, horizontal flip, vertical
flip, rotation, and translation. Rotation and translation use nearest-neighbor
sampling and fill coordinates outside the image with zero. Resize also uses
nearest-neighbor sampling and preserves the channel count. Random rotation,
translation, resize, and vertical flipping are disabled by default.

Blur is applied after spatial transforms and uses a separable odd-sized
Gaussian kernel with zero padding. Gaussian noise is then added without
clipping, followed by brightness, contrast, and per-channel normalization.
All image operations support `[height, width]` and `[channels, height, width]`
samples and use the configured seeded generator.

Horizontal image flipping is disabled by default. The seeded constructor can
enable it with a probability between `0.0` and `1.0`:

```cpp
DataLoader augmented_loader(
    dataset,
    batch_size,
    max_samples,
    true,
    BatchFormat::CNN,
    12345,
    0.5
);
```

Flipping supports `[height, width]` and `[channels, height, width]` samples,
preserves labels, and uses the loader seed for reproducible augmentation.

Random cropping is also available through the same constructor by providing
positive crop height and width after the flip probability:

```cpp
DataLoader cropped_loader(
    dataset,
    batch_size,
    max_samples,
    true,
    BatchFormat::CNN,
    12345,
    0.5,
    24,
    24
);
```

The crop must fit within every image. Crop coordinates are sampled
independently for each sample, and the same seeded generator controls both
cropping and flipping. Leaving both crop dimensions at `0` disables cropping
and preserves the original sample dimensions.

Brightness and contrast adjustments are disabled by default. They can be
enabled after the crop parameters with a probability and positive factor
range:

```cpp
DataLoader photometric_loader(
    dataset,
    batch_size,
    max_samples,
    true,
    BatchFormat::CNN,
    12345,
    0.5,
    0,
    0,
    {},
    {},
    0.5,
    0.8,
    1.2,
    0.5,
    0.8,
    1.2
);
```

Contrast is applied around each sample channel's mean, after brightness and
before channel normalization. This definition works for arbitrary numeric
pixel ranges and does not clip values. The same seeded generator controls all
random augmentations, while labels and output shapes are preserved.

## CNN example

The complete CNN example is in
[examples/fashion_mnist_cnn.cpp](examples/fashion_mnist_cnn.cpp).

Its architecture is:

```text
Conv2D(1, 8, 3, 3)
ReLU
MaxPool2D(2, 2, stride 2)
Flatten
Linear(1352, 10)
```

The flattened size is 1352 because:

```text
28 x 28
Conv2D with 3 x 3 kernel -> 26 x 26
MaxPool2D with 2 x 2 kernel and stride 2 -> 13 x 13
8 channels x 13 x 13 = 1352
```

This architecture was kept intentionally small. It is large enough to
exercise convolution, pooling, flattening, classification, and training
without introducing additional layers that would obscure the execution
changes being studied.

## Reproducible benchmarks and validation

There is currently no formal build system. The following commands show the
source files used by the main CNN benchmark.

Build without OpenMP:

```bash
g++ -std=c++17 -O2 -Iinclude \
  examples/fashion_mnist_cnn_benchmark.cpp \
  src/activation.cpp src/conv2d.cpp src/cross_entropy.cpp \
  src/dataloader.cpp src/dataset.cpp src/execution.cpp \
  src/flatten.cpp src/idx_reader.cpp src/initialization.cpp \
  src/linear.cpp src/maxpool2d.cpp src/relu.cpp src/softmax.cpp \
  src/tensor.cpp src/tensor_ops.cpp src/trainer.cpp \
  -o fashion_mnist_cnn_benchmark_serial
```

Build with OpenMP:

```bash
g++ -std=c++17 -O2 -fopenmp -Iinclude \
  examples/fashion_mnist_cnn_benchmark.cpp \
  src/activation.cpp src/conv2d.cpp src/cross_entropy.cpp \
  src/dataloader.cpp src/dataset.cpp src/execution.cpp \
  src/flatten.cpp src/idx_reader.cpp src/initialization.cpp \
  src/linear.cpp src/maxpool2d.cpp src/relu.cpp src/softmax.cpp \
  src/tensor.cpp src/tensor_ops.cpp src/trainer.cpp \
  -o fashion_mnist_cnn_benchmark
```

Run the benchmark from the repository root after the raw Fashion-MNIST files
are available under `data/FashionMNIST/raw/`.

The benchmark uses the same architecture, dataset, batch size, epoch count,
and learning rate for both modes. Initial parameters are captured once and
copied into the serial and OpenMP models. A one-batch warm-up is performed
before timing. Evaluation uses the same non-shuffled DataLoader path in both
modes.

One completed run produced:

| Mode | Total training time | Average batch time | Training loss | Test accuracy |
|---|---:|---:|---:|---:|
| Serial | 233.758 seconds | 0.249209 seconds | 0.638432 | 0.764100 |
| OpenMP, 4 threads | 102.054 seconds | 0.108799 seconds | 0.642045 | 0.751300 |

The measured speedup was:

```text
233.758 / 102.054 = 2.290543x
```

The small metric difference is expected because the trainer uses shuffled
training batches and parallel floating-point reductions can have different
execution order. The benchmark is primarily a performance measurement; the
consistency test described below checks numerical differences directly.

The focused validation example is
[examples/cnn_serial_parallel_consistency_test.cpp](examples/cnn_serial_parallel_consistency_test.cpp).
It creates identical models, inputs, and labels, then compares:

- Forward outputs.
- Loss.
- Conv2D `grad_bias`.
- Conv2D `grad_weight`.
- Conv2D `dX`.
- Updated parameters.

It tests batch sizes 2 and 8. In the OpenMP build, the largest observed
Conv2D `dX` difference was approximately:

```text
maximum absolute difference: 6.94e-17
maximum relative difference: 1.60e-13
```

All other reported comparisons were zero in that run. The tiny `dX`
difference comes from floating-point addition order during private-buffer
reduction. The validation tolerance is `1e-10`.

Other focused examples cover:

- Conv2D forward behavior and numerical gradients.
- MaxPool2D forward, backward, and numerical gradients.
- ReLU and Flatten serial-versus-parallel behavior.
- Matrix multiplication with rectangular and non-contiguous inputs.
- MLP and CNN Trainer batch formats.
- IDX parsing and validation.

## Temporary memory tradeoffs

Private gradient buffers make overlapping backward operations race-free, but
they consume additional memory.

For Conv2D backward, the parallel `dX` path uses approximately:

```text
number_of_worker_buffers x dX.size() x sizeof(double)
```

The current implementation sizes the buffers from the OpenMP maximum thread
count. These buffers are allocated only when OpenMP is available and the
output workload reaches the shared threshold. Serial and below-threshold
paths do not allocate them.

This choice favors correctness and simple ownership. A future optimization
could reuse workspace buffers or size them from the actual active worker
count, but that should be measured before adding complexity.

## Repository structure

```text
NeuroFroge/
├── include/        Public headers and interfaces
├── src/            Tensor operations, layers, data, execution, and training
├── examples/       Executable examples, benchmarks, and focused tests
├── data/           Local datasets; ignored by Git
├── main.cpp        Original project entry point
└── README.md       Project documentation
```

Important headers include:

- `tensor.h`: tensor storage, shape, strides, views, and indexing.
- `tensor_ops.h`: matrix and tensor operations.
- `module.h`: common layer interface.
- `sequential.h`: model composition.
- `conv2d.h`, `maxpool2d.h`, `flatten.h`: CNN layers.
- `dataloader.h`, `dataset.h`, `idx_reader.h`: data handling.
- `trainer.h`: mini-batch training.
- `execution.h`: shared CPU execution policy.

## Remaining work

The current roadmap is:

1. Continue improving tensor-level CPU execution and measure each change.
2. Add a formal build and test configuration.
3. Improve workspace reuse for private gradient buffers.
4. Expand data pipeline and checkpoint support.
5. Add additional CNN layers and model architectures.
6. Explore GPU execution after CPU behavior is well understood.
7. Implement embeddings, normalization, attention, and Transformer blocks.

Each future optimization should preserve a serial reference path, define its
memory ownership rules, include numerical validation, and report its
performance tradeoffs.

## Development philosophy

NeuroFroge follows an incremental approach:

1. Implement a small, inspectable component.
2. Validate its numerical behavior.
3. Test contiguous and non-contiguous data where applicable.
4. Add parallel execution only after ownership and reduction rules are clear.
5. Compare against the serial reference.
6. Measure performance before deciding whether additional complexity is
   justified.

The project is built to make the reasoning behind each implementation
decision visible, not just to produce a working model.
