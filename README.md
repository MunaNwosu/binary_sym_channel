# Neural Network with BSC (Binary Symmetric Channel) Simulation

This repository contains a PyTorch implementation of a simple neural network model trained on the MNIST dataset. The project simulates the transmission of model parameters (weights) over a Binary Symmetric Channel (BSC) with different error probabilities, and evaluates the model's performance before and after transmission.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Overview

This project consists of the following main components:

1. **Neural Network Model**:
   - A fully connected neural network with two hidden layers and ReLU activations.
   - Dropout is applied during training to reduce overfitting.
   - The output layer is used for 10-class classification (MNIST digits 0–9).

2. **Binary Symmetric Channel (BSC) Simulation**:
   - Simulates the transmission of model parameters over a noisy communication channel by adding Gaussian noise to the model weights.
   - The simulation tests the model’s robustness to different error probabilities in the range: \(10^{-6}, 10^{-4}, 10^{-2}, 10^{-1}\).

3. **Training and Testing**:
   - The model is trained on the MNIST training set using the Adam optimizer and cross-entropy loss function.
   - The accuracy is evaluated on the test set before and after the model weights are transmitted over the BSC.

## Requirements

To run this project, you need:

- Python 3.x
- PyTorch
- torchvision

You can install the required packages using `pip`:

```bash
pip install torch torchvision
```

## Training and Evaluation
Training: The network is trained for 10 epochs using the Adam optimizer with a learning rate of 0.001. The loss is calculated using the cross-entropy loss function.
Evaluation: After training, the model's accuracy is evaluated on the MNIST test set. The model parameters are then transmitted over the BSC with error probabilities of 
10^-6,10^-4,10^-2,10^-1, and the accuracy is evaluated again.

## Results
The results will show the model’s performance on the MNIST test set before and after transmission over the Binary Symmetric Channel. You will observe the impact of different error probabilities on the model's accuracy.
