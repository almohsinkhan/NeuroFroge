# NeuroFroge

**A neural network framework built from scratch in C++.**

NeuroFroge is a learning project focused on understanding how deep learning frameworks work internally. Instead of relying on high-level libraries, I'm implementing the fundamental building blocks of neural networks from scratch in C++.

The long-term goal is to build a Transformer from scratch. Rather than jumping directly into Transformers, I'm taking an incremental approach: understanding tensors, backpropagation, memory management, and neural network architectures before moving toward CNNs, GPU acceleration, and attention mechanisms.

## Current Progress

NeuroFroge currently supports the basic workflow required to train a neural network.

- **Tensor operations:** N-dimensional tensors, indexing, broadcasting, reshaping, and transposing.
- **Memory management:** Shared-storage views, stride-based indexing, and contiguous-memory detection.
- **Matrix operations:** Matrix multiplication and specialized multiplication operations used during backpropagation.
- **Neural network layers:** Linear layers and activation functions, including ReLU, Sigmoid, Tanh, and Softmax.
- **Training:** Cross-entropy loss, backpropagation, gradient accumulation, SGD, and mini-batch training.
- **Data handling:** Loading image and label data from CSV files.

I'm also experimenting with CPU performance through profiling, compiler optimizations, and more efficient tensor access.

## Current Model

The current experiment uses a fully connected neural network trained on Fashion-MNIST.

| Component | Configuration |
|---|---|
| Input | 784 features (28 × 28 image) |
| Hidden layer | Linear: 784 → 128 |
| Activation | ReLU |
| Output layer | Linear: 128 → 10 |
| Output activation | Softmax |
| Loss | Cross-entropy |
| Optimizer | SGD |

The model classifies images into the 10 Fashion-MNIST categories.

Training and evaluation are still experimental. Accuracy varies between runs, so I plan to establish a reproducible benchmark as the framework develops.

## How Training Works

The framework implements the main steps of a neural network training loop:

1. Load a batch of images and labels.
2. Perform the forward pass to generate predictions.
3. Calculate the loss by comparing predictions with the targets.
4. Propagate gradients backward through the network.
5. Accumulate gradients for the trainable parameters.
6. Update the parameters using gradient descent.

Implementing these steps myself helps me understand how data and gradients move through a neural network, rather than treating training as a black box.

## Project Structure

```text
NeuroFroge/
├── include/        # Header files and interfaces
├── src/            # Tensor operations, layers, losses, and dataset
├── examples/       # Small experiments and tests
├── data/           # Local training data
├── main.cpp        # Training entry point
└── README.md
```

The implementation is organized into separate modules so that individual components can be developed, tested, and improved independently.

## Roadmap

### 1. Tensor and Neural Network Fundamentals

- [x] N-dimensional Tensor class
- [x] Tensor indexing and operations
- [x] Matrix multiplication
- [x] Linear layer and weight initialization
- [x] Activation and loss functions
- [x] Backpropagation and gradient accumulation
- [x] SGD and mini-batch training
- [x] CSV dataset loading
- [x] Broadcasting and stride-based views
- [x] Reshape and transpose views
- [x] Initial CPU performance optimizations

### 2. Convolutional Neural Networks

- [ ] Implement Conv2D
- [ ] Implement the Conv2D backward pass
- [ ] Implement max pooling
- [ ] Implement a flatten layer
- [ ] Build a CNN
- [ ] Train and evaluate it on Fashion-MNIST

### 3. Performance and Systems Engineering

- [ ] Improve matrix multiplication further
- [ ] Add multithreaded CPU operations
- [ ] Improve memory efficiency
- [ ] Introduce a proper data pipeline
- [ ] Explore CUDA and GPU tensor operations

### 4. Transformers

- [ ] Token embeddings
- [ ] Positional encoding
- [ ] Layer normalization
- [ ] Multi-head self-attention
- [ ] Feed-forward network
- [ ] Residual connections
- [ ] Transformer block
- [ ] Training pipeline
- [ ] Build and train a complete Transformer

## Why NeuroFroge?

The goal isn't to compete with mature frameworks such as PyTorch. It's to understand the engineering and mathematics behind them.

Through this project, I want to explore:

- How tensors are represented and accessed in memory.
- How matrix multiplication and broadcasting work.
- How layers store parameters and calculate gradients.
- How backpropagation and optimizers work internally.
- How convolutional operations can be implemented efficiently.
- How CPU and GPU execution differ.
- How these components eventually come together in a Transformer.

Each feature is an opportunity to experiment, measure performance, investigate bugs, and understand the underlying implementation.

## Development Philosophy

**Build it. Break it. Understand it. Improve it.**

NeuroFroge is being developed incrementally, with an emphasis on understanding each component before moving to the next. The aim is not simply to implement a neural network, but to learn how deep learning systems are built from the ground up.