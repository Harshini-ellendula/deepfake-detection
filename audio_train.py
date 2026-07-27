import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X = np.load("audio_features.npy")
y = np.load("audio_labels.npy")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Neural Network
class AudioClassifier(nn.Module):

    def __init__(self):
        super(AudioClassifier, self).__init__()

        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = AudioClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Audio Deepfake Model...\n")

# Training loop
for epoch in range(20):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/20  |  Loss: {loss.item():.4f}")


# Testing
with torch.no_grad():

    predictions = model(X_test)
    _, predicted = torch.max(predictions, 1)

    acc = accuracy_score(y_test.numpy(), predicted.numpy())

print("\nTest Accuracy:", acc * 100, "%")

# Save model
torch.save(model.state_dict(), "audio_model.pth")

print("\nAudio model saved as audio_model.pth")