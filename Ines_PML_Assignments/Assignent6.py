import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import numpy as np
from torchsummary import summary

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Reshape function to match NCHW format
def reshape_cifar(X):
    return X.transpose(0, 3, 1, 2)  # Convert to NCHW format

# Set CIFAR-10 parameters
batch_size = 64
channels = 3
width = 32
height = 32

# Reshape dataset
X_train = reshape_cifar(trainset.data)
y_train = np.array(trainset.targets)
X_test = reshape_cifar(testset.data)
y_test = np.array(testset.targets)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, width * height * channels)).reshape(-1, channels, width, height)
X_test = scaler.transform(X_test.reshape(-1, width * height * channels)).reshape(-1, channels, width, height)

# Convert numpy arrays to PyTorch tensors with right shape
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Instantiate the model
hidden_size = 8
output_size = 10
model = nn.Sequential(
    nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(8 * 16 * 16, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(hidden_size, output_size)
)

# Print model summary
summary(model, (channels, width, height))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dataloaders
train_dl = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_dl = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Train the model and predict on test samples to estimate accuracy
def train(model, optimizer, loss_fn, num_epochs, train_dl, test_dl):
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train_dl.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        model.eval()
        running_test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_test_loss = running_test_loss / len(test_dl.dataset)
            history['test_loss'].append(epoch_test_loss)
            test_accuracy = correct / total * 100
            history['test_accuracy'].append(test_accuracy)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    return history

# Train the model and plot accuracy
history = train(model, optimizer, criterion, 5, train_dl, test_dl)

# Plot accuracy
plt.plot(history['test_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()
plt.show()
