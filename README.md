# LeNet5 Reimplementation in PyTorch

This is a faithful reimplementation of the original LeNet5 architecture from the paper "Gradient-based learning applied to document recognition" (LeCun et al., 1998), trained on the MNIST dataset.

## Overview
This reimplementation reproduces key features of the original architecture:
- Input images resized to 32x32 and scaled to [-0.1, 1.175]
- Convolutional and subsampling layers arranged in a "bipyramid"
- tanh activations
- Subsampling layers with learnable weights and biases
- MSE loss with one-hot encoded labels
- LeCun normal initialization for all layers

## Results
| Metric | Value |
|--------|--------|
| Training Accuracy | ~98% |
| Test Accuracy | ~98% |

Example predictions, training curves, and a confusion matrix are in the 'results/' folder.

## How to Run
```bash
git clone https://github.com/yourusername/lenet5-replication
cd lenet5-replication
pip install -r requirements.txt
python lenet5.py
'''

