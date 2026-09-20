# NeuroFroge

This project is a fun learning project where I am implementing everything from scratch in C++. I am building this project to understand what is happening deep down behind big frameworks like NumPy, scikit-learn, and PyTorch.

I have implemented a Tensor class for now, which is good enough to start implementing a neural network. I still need to improve it because I am thinking of eventually building an LLM using this whole setup.

## What we have done so far

### Tensor

* Element-wise operations on tensors: addition, subtraction, and multiplication
* 2D matrix multiplication
* Sum of elements
* Mean of elements

## Next

* Implement activation functions

There is still a long way to go.


## backpropagation
** for BCE + sigmoid together
![alt text](<Screenshot From 2026-09-15 15-56-14.png>)

## performance of architecture linear -> relu -> linear -> sigmoid 

Epoch 0 Loss: 0.706497
Epoch 1000 Loss: 0.00470452
Epoch 2000 Loss: 0.00205834
Epoch 3000 Loss: 0.00130318
Epoch 4000 Loss: 0.000948606
Epoch 5000 Loss: 0.00074417
Epoch 6000 Loss: 0.000611145
Epoch 7000 Loss: 0.000518016
Epoch 8000 Loss: 0.000449249
Epoch 9000 Loss: 0.000396316

XOR Predictions
0 XOR 0 = 0.00061528
0 XOR 1 = 0.999907
1 XOR 0 = 0.999907
1 XOR 1 = 0.00061528




