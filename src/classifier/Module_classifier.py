import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

class SimpleLinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_layer_dim,act_fun=torch.relu,dropout_prob=0.3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.bn1 = nn.BatchNorm1d(hidden_layer_dim)
        self.fch = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.bn2 = nn.BatchNorm1d(hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

        self.act_fun = act_fun
        self.dropout = nn.Dropout(dropout_prob)

        torch.nn.init.eye_(self.fch.weight)

        self.layers = [self.fc1, self.fch, self.fc2]

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        x = self.fch(x)
        x = self.bn2(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)
        return x

def train(net, data, labels, lr=0.001, max_num_epochs=1000, batch_size=64, patience=50,verbose=True):
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    data = torch.tensor(data,dtype=torch.float32)
    labels = torch.tensor(labels,dtype=torch.int64)
    
    dataset = TensorDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_num_epochs):
        net.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            optim.zero_grad()
            outputs = net(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len_train_loader

        # check for convergence for early stopping
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                outputs = net(val_data)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
        val_loss /= len_val_loader
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if verbose:
            print(f'\rEpoch {epoch+1}, Training Loss: {train_loss:.5g}, Validation Loss: {val_loss:.5g}', end='', flush=True)
        if patience_counter >= patience:
            if verbose:
                print('\nEarly stopping triggered')
            break


def compute_classification_acc(data,labels,test_data,test_labels,linerClassifier,output_dim,hidden_layer_dim=-1,verbose=True):
    input_dim = data.shape[1]
    if linerClassifier:
        Cnet = SimpleLinearClassifier(input_dim,output_dim)
        hidden_layer_dim = -1
    else:
        if hidden_layer_dim==-1:
            hidden_layer_dim=output_dim
        Cnet = Classifier(input_dim,output_dim,hidden_layer_dim)

    train(Cnet,data,labels,verbose=verbose)

    class_probabilities = Cnet(torch.tensor(data,dtype=torch.float32))
    c_labels = torch.argmax(class_probabilities,dim=1)
    labels = torch.tensor(labels,dtype=torch.int64)
    train_accuracy = (labels==c_labels).int().count_nonzero()/labels.numel()

    test_class_probabilities = Cnet(torch.tensor(test_data,dtype=torch.float32))
    test_c_labels = torch.argmax(test_class_probabilities,dim=1)
    test_labels = torch.tensor(test_labels,dtype=torch.int64)
    test_accuracy = (test_labels==test_c_labels).int().count_nonzero()/test_labels.numel()

    return train_accuracy, test_accuracy


import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(net, x_min, x_max, y_min, y_max, h=0.01, data=None, labels=None):
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Predict class probabilities for each point in the grid
    with torch.no_grad():
        outputs = net(grid_tensor)
        predictions = outputs.argmax(dim=1).numpy()
    predictions = predictions.reshape(xx.shape)
    
    # Plot the heatmap
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.Spectral)
    
    # Plot the data points
    if data is not None and labels is not None:
        scatter = plt.scatter(data[:, 0], data[:, 1], s=8, c=labels, edgecolor='black', cmap=plt.cm.Spectral)
    
    cbar = plt.colorbar(scatter, ticks=np.arange(len(np.unique(labels))))
    cbar.set_label('Class Labels')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary')
    plt.show()
