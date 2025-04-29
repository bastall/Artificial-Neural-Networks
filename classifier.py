import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os

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

def evaluate_model (model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy        

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cpu')  # use CPU
    model = model.to(device)
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training log
    log_entries = []
    best_accuracy = 0.0
    total_step = len(train_loader)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
         
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backwrad and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print step statistics
            if (i+1) % 50 == 0:
                log_entry = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, i+1, total_step, loss.item())
                print(log_entry)
                log_entries.append(log_entry)
            
        # Evaluate model after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        log_entry = 'Test Accuracy: {:.2f} %'.format(accuracy)
        print(log_entry)
        log_entries.append(log_entry)
                
        # Save if it's the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_fashion_model.pth')
        
    print(f'Best accuracy: {best_accuracy:.2f}%')
    print(f'Training complete!')
    
    # Save final model
    torch.save(model.state_dict(), 'final_fashion_model.pth')
    
    # Save training log to file
    with open('log.txt', 'w') as f:
        for entry in log_entries:
            f.write(entry + '\n')
    
    return model

def predict_image(model, image_path):
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    device = torch.device('cpu')  # use CPU
    model = model.to(device)
    model.eval()
    
    try:
        # Load grayscale image using torchvision.io 
        img = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.GRAY)
        img = img.squeeze() 
        
        # Resize to 28x28 if needed
        if img.shape != torch.Size([28, 28]):
            transform_resize = transforms.Resize((28, 28), antialias=True)
            img = transform_resize(img.unsqueeze(0)).squeeze(0)
            
         # Apply transformations
        transform_norm = transforms.Normalize((0.5,), (0.5,))
        img_tensor = transform_norm(img.unsqueeze(0)).unsqueeze(0)  # Add channel and batch dimensions
        img_tensor = img_tensor.to(device) 
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
         # Get prediction
        predicted_class = class_labels[predicted.item()]
        return predicted_class
        
    except Exception as e:
        return f"Error processing image: {str(e)}"
    
    
def load_or_train_model(model, train_loader, test_loader):
    if os.path.exists('best_fashion_model.pth'):
        model.load_state_dict(torch.load('best_fashion_model.pth', map_location=torch.device('cpu')))
        print("Loaded pre-trained model")
    else:
        print("Training new model...")
        model, training_log = train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001)
        # Save training log
        with open('log.txt', 'w') as f:
            for entry in training_log:
                f.write(f"Epoch: {entry['epoch']}, Loss: {entry['loss']:.4f}, Accuracy: {entry['accuracy']:.2f}%\n")
    return model

def interactive_predict(model):
    print("Done!")
    while True:
        filepath = input("Please enter a filepath:\n")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break
        prediction = predict_image(model, filepath)
        print(f"Classifier: {prediction}")

def main():
    # Load data
    train_loader, test_loader, _, _ = load_data(batch_size=64)
    
    # Check if a trained model exists, if not, train one
    model = FashionClassifier()
    
    model = load_or_train_model(model, train_loader, test_loader)
    
    # Interactive loop for predictions
    interactive_predict(model)


if __name__ == "__main__":
    main()
