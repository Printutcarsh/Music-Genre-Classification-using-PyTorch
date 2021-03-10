import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_PATH = 'data.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First we will load the data
def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert list into numpy arrays
    inputs = torch.tensor(data['mfcc'])
    targets = torch.tensor(data['labels'])

    return inputs, targets

def prepare_dataset(test_size, validation_size):
    # Load the data
    X, y = load_data(DATASET_PATH)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=743)

    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=743)

    # In CNN the input must contain channel also
    # So we are reshaping the data
    X_train = torch.unsqueeze(X_train, 1)
    X_test = torch.unsqueeze(X_test, 1)
    X_val = torch.unsqueeze(X_val, 1)

    # Creating tensor dataset
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    val = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    return train_loader, test_loader, val_loader

# Create model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(31*2*32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn(self.pool(F.relu(self.conv1(x))))
        x = self.bn(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 31*2*32)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

model = CNN().to(device)

train_loader, test_loader, val_loader = prepare_dataset(0.25, 0.2)

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 30

print("Training Start")

for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader)) # To get progress bar
    for i, (inputs, labels) in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = running_loss)

    # Evaluating on validation set
    with torch.no_grad():
        print("Evaluate")
        model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            num_samples = labels.size(0)
            val_acc = running_corrects.double() / num_samples
            print(f"Validation Accuracy {val_acc :.4f}")

    model.train()

# Make predicton on a sample from test set
def predict(data):
    X, y = next(iter(data))
    X = X.cuda()
    y = y.cuda()
    acc = 0
    with torch.no_grad():
        output = model(X)
        _, preds = torch.max(output, 1)
        print("Expected index: {} Predicted index {}".format(y.item(), preds.item()))

predict(test_loader)
