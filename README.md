# Java CNN

A Convolutional Neural Network built from scratch in Java with no ML libraries, trained on the MNIST handwritten digit dataset.

## Architecture
- **Input:** 28×28 grayscale images
- **Conv Layer:** 8 filters, 5×5 kernel, stride 1, ReLU
- **Max Pool Layer:** 2×2 window, stride 2
- **Fully Connected Layer:** 1152 → 10 neurons, Softmax

## How it works
- Forward pass: convolution → pooling → fully connected → softmax
- Backward pass: full SGD with cross-entropy loss
- No external ML libraries. Everything implemented from scratch

## Setup
1. Download MNIST CSV data from https://pjreddie.com/projects/mnist-in-csv/
2. Place `mnist_train.csv` and `mnist_test.csv` in the `data/` folder
3. Run `Main.java`

## Background
This project was built to solidify concepts from my NPTEL Deep Learning certification by implementing a CNN entirely from scratch in Java.

## Results
Trained on 3,000 MNIST images for 1 epoch (quick test run).
With the full 60,000 training images over 3 epochs, expected accuracy is 95-97%.

Epoch 1 — avg loss: 0.1336 | train accuracy: 95.93%
Epoch 2 — avg loss: 0.0649 | train accuracy: 97.99%
Epoch 3 — avg loss: 0.0508 | train accuracy: 98.40%
Test accuracy: 98.00% (980/1000)

No overfitting observed.
