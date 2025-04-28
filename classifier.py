import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        
        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size = 2)
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # to help with preventing overfitting
            nn.Linear(512, 10) 
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1,1]
    ])
    
    # Load training data
    train_dataset = datasets.FashionMNIST(
        root = ".",
        train = True,
        transform = transform,
        download = False
    )
    
    # Load test data
    test_dataset = datasets.FashionMNIST(
        root = ".",
        train = False,
        transform = transform,
        download = False
    )
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    ) 
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

# Test if data is loading correctly
#def test_data_loading(train_loader):
    # Get a single batch from the train_loader
  #  data_iter = iter(train_loader)
   # images, labels = next(data_iter)

    # Print the shape of the images and the corresponding labels
    #print(f"Batch of images shape: {images.shape}")
    #print(f"Batch of labels shape: {labels.shape}")

    # visualize one image from the batch
    #plt.imshow(images[0].squeeze(), cmap='gray')
    #plt.title(f"Label: {labels[0]}")
    #plt.show()

#train_loader, test_loader, _, _ = load_data()
#test_data_loading(train_loader)

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_vailable else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training log
    log = []
    best_accuracy = 0.0