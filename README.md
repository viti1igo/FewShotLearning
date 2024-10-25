# LOG: Studying ML

This is my attempt to learn the Few Shot Learning concept. This project implements a few-shot learning framework using a Prototypical Network on the Omniglot dataset. The Prototypical Network learns to classify samples based on their proximity to class prototypes in an embedding space, enabling efficient few-shot learning.
## Introduction
Few-shot learning refers to the task of classifying new classes with only a few labelled examples. This project uses a Prototypical Network, which computes a prototype (mean of the support set embeddings) for each class in the embedding space. During inference, new examples are classified by finding the nearest prototype based on a distance metric (e.g., Euclidean distance).
Core Concepts:
Prototypical Network: Learns a metric space where computing distances can perform classification to prototypes of each class.
Few-Shot Learning: A model’s ability to generalize to new classes using very few labelled examples.
## Learning objectives
- Get to know PyTorch library.
- Import dataset from torchvision library
- Applying transfer learning to use the model which is already pre-built and trained.
- Learn the ideas of Few-Shot Learning using the concept of Prototypical Network model.
## Dataset
The Omniglot dataset consists of 20 instances for each of the 1,623 characters from 50 different alphabets. It is often used for evaluating few-shot learning models. You can download it directly using Torchvision:
```python
from torchvision.datasets import Omniglot
train_dataset = Omniglot(root='data/', download=True, transform=your_transform)
```
## Citation
Bennequin, E. easyfsl [Computer software]. https://github.com/sicara/easy-few-shot-learning

Snell, J., Swersky, K., & Zemel, R. S. (2017). Prototypical networks for few-shot learning. arXiv. https://arxiv.org/abs/1703.05175
