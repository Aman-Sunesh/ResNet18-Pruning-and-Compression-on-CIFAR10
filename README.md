# ResNet18 Pruning and Compression on CIFAR-10

This repository documents a PyTorch pruning study for ResNet-18 on CIFAR-10. The project compares several neural network pruning strategies and analyzes the tradeoff between model sparsity, accuracy, and practical deployability.

## Overview

The project explores four pruning workflows:

- **Model evaluation:** accuracy, sparsity, MACs, and parameter count using a pretrained ResNet-18.
- **Unstructured magnitude pruning:** layer-wise pruning based on absolute weight magnitude.
- **Regularization-assisted pruning:** L1/L2 fine-tuning followed by sensitivity analysis.
- **Structured filter pruning:** filter-wise pruning using per-filter L1 norm.
- **Post-pruning retraining:** mask-based retraining to preserve pruned weights while recovering accuracy.

## Model and Dataset

- **Model:** ResNet-18
- **Dataset:** CIFAR-10
- **Framework:** PyTorch
- **Hardware:** CUDA-enabled GPU
- **Profiling:** THOP for MACs and parameter count

## Key Results

| Experiment | Result |
|---|---:|
| Baseline accuracy | 93.79% |
| Baseline MACs | 557.880M |
| Baseline parameters | 11.174M |
| Baseline sparsity | 0.00% |
| Layer-wise magnitude pruning sparsity | 47.82% |
| Layer-wise magnitude pruning accuracy | 93.42% |
| Structured filter pruning accuracy | 88.83% |
| Structured pruning + retraining accuracy | 93.82% |

## Main Takeaways

Magnitude pruning works well when applied layer-wise because different ResNet layers have different weight scales and pruning sensitivity. L1 regularization generally improves pruning robustness by pushing more weights near zero, making magnitude-based removal safer. Structured filter pruning is more hardware-friendly but initially causes a larger accuracy drop, which can be largely recovered through masked retraining.

## Technologies

Python, PyTorch, TorchVision, CUDA, THOP, NumPy, Matplotlib, tqdm

## Possible Extensions

- Compare pruning behavior across ResNet-34 or MobileNet.
- Measure real inference latency before and after structured pruning.
- Export pruned models to ONNX or TorchScript.
- Test pruning-aware fine-tuning schedules with different regularization strengths.
