# NeuroFroge

A neural network framework built from scratch in **C++**, with the goal of understanding how deep learning systems work internally by implementing the fundamental components without relying on high-level frameworks.

NeuroFroge started with a simple goal: eventually build a **Transformer from scratch**. Instead of jumping directly into Transformers, I'm building the framework step by step — starting from tensors and backpropagation, then moving toward CNNs, GPU acceleration, and eventually Transformers.

## Current Progress

### Core Components

* [x] Tensor class
* [x] N-dimensional tensor representation
* [x] Tensor indexing and operations
* [x] Matrix multiplication
* [x] Linear layer
* [x] Weight initialization
* [x] ReLU
* [x] Sigmoid
* [x] Tanh
* [x] Softmax
* [x] Cross-entropy loss
* [x] Backpropagation
* [x] Gradient accumulation
* [x] SGD
* [x] Mini-batch training
* [x] Dataset loading from CSV

### Current Network

The current model is:

```text
Input
784
 │
 ▼
Linear
784 → 128
 │
 ▼
ReLU
 │
 ▼
Linear
128 → 10
 │
 ▼
Softmax
 │
 ▼
Prediction
```

It is currently being trained on **Fashion-MNIST**.

With 800 training samples and 200 test samples, the current implementation reaches approximately:

```text
Test Accuracy: ~79%
```

This is mainly an experimental implementation intended to validate the framework and understand the underlying mechanics.

## Training Pipeline

The complete training pipeline currently looks like:

```text
Dataset
   │
   ▼
Tensor
   │
   ▼
Forward Pass
   │
   ▼
Loss
   │
   ▼
Backpropagation
   │
   ▼
Gradient Accumulation
   │
   ▼
Gradient Descent
   │
   ▼
Updated Parameters
```

Mini-batch training is currently supported:

```text
Batch of images
      │
      ▼
Forward Pass
      │
      ▼
Calculate Loss
      │
      ▼
Backward Pass
      │
      ▼
Accumulate Gradients
      │
      ▼
Update Parameters
```

## Project Structure

```text
NeuroFroge/
│
├── include/
│   ├── tensor.h
│   ├── tensor_ops.h
│   ├── linear.h
│   ├── activation.h
│   ├── softmax.h
│   ├── cross_entropy.h
│   ├── initialization.h
│   └── dataset.h
│
├── src/
│   ├── tensor.cpp
│   ├── tensor_ops.cpp
│   ├── linear.cpp
│   ├── activation.cpp
│   ├── softmax.cpp
│   ├── cross_entropy.cpp
│   ├── initialization.cpp
│   └── dataset.cpp
│
├── data/
│   ├── images.csv
│   └── labels.csv
│
├── main.cpp
└── README.md
```

## Dataset

The current experiment uses **Fashion-MNIST**.

Images are:

```text
28 × 28
```

and flattened into:

```text
784 × 1
```

The 10 output classes correspond to the Fashion-MNIST categories.

The dataset is currently preprocessed using Python and exported as CSV so that the C++ framework can load it directly.

## Why Build This?

The goal isn't to compete with frameworks such as PyTorch.

The goal is to understand what happens underneath them.

I want to understand things like:

* How tensors are represented in memory
* How matrix multiplication works
* How layers store parameters
* How gradients flow backward
* How optimizers update parameters
* How convolution works
* How GPU computation can accelerate these operations
* How the components eventually fit together to build a Transformer

## Roadmap

### Phase 1 — Neural Network Fundamentals

* [x] Tensor
* [x] Linear layer
* [x] Activations
* [x] Softmax
* [x] Loss functions
* [x] Backpropagation
* [x] SGD
* [x] Mini-batch training

### Phase 2 — Computer Vision

* [ ] Conv2D
* [ ] Conv2D backward pass
* [ ] Max Pooling
* [ ] Flatten layer
* [ ] CNN
* [ ] Train CNN on Fashion-MNIST

### Phase 3 — Performance

* [ ] Improve CPU utilization
* [ ] Multithreaded operations
* [ ] Optimize matrix multiplication
* [ ] Memory optimization
* [ ] CUDA support
* [ ] GPU tensor operations

### Phase 4 — Transformer

* [ ] Embeddings
* [ ] Positional encoding
* [ ] Layer normalization
* [ ] Multi-head self-attention
* [ ] Feed-forward network
* [ ] Residual connections
* [ ] Transformer block
* [ ] Training pipeline
* [ ] Complete Transformer

## Current Focus

The immediate goal is to implement:

```text
Conv2D
   ↓
Backward Pass
   ↓
Pooling
   ↓
CNN
```

After building and understanding the CNN, I'll continue toward GPU acceleration and eventually return to the original goal:

```text
Tensor
   ↓
Neural Networks
   ↓
CNN
   ↓
GPU
   ↓
Attention
   ↓
Transformer
```

## Philosophy

**Build it. Break it. Understand it. Improve it.**

NeuroFroge is primarily a learning project, where every component is implemented to understand what is happening underneath modern deep learning frameworks.
