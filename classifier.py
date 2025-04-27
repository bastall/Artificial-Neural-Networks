import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1,1]
    ])
    
    train_dataset = datasets.FashionMNIST(
        root = ".",
        train = True,
        transform = transform,
        download = False
    )