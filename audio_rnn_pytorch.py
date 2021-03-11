import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_PATH = 'data.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# First we will load the data
def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists into tensors
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

    # Creating tensor dataset
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    val = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    return train_loader, test_loader, val_loader

# Create model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        return out

model = LSTM(input_size=13, hidden_size=64, num_layers=2, num_classes=10).to(device)

train_loader, test_loader, val_loader = prepare_dataset(0.25, 0.2)

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 30

for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0
    loop = tqdm(train_loader)
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = running_loss/len(train_loader))

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
            val_acc = running_corrects.double()/num_samples
            print(f"Validation Accuracy {val_acc :.4f}")

    model.train()

# Make predicton on a sample from test set
def predict(data):
    X, y = next(iter(data))
    X = X.cuda()
    y = y.cuda()
    with torch.no_grad():
        output = model(X)
        _, preds = torch.max(output, 1)
        print("Expected index: {} Predicted index {}".format(y.item(), preds.item()))

predict(test_loader)
