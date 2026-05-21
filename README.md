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
- No external ML libraries — everything implemented from scratch

## Setup
1. Download MNIST CSV data from https://pjreddie.com/projects/mnist-in-csv/
2. Place `mnist_train.csv` and `mnist_test.csv` in the `data/` folder
3. Run `Main.java`
