import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data_and_plots import saveTotalLossPlots

class IndexDataset(Dataset):
    def __init__(self, distance_matrix):

        self.distance_matrix = distance_matrix

    def __len__(self):
        return self.distance_matrix.shape[0]

    def __getitem__(self, idx):
        # One-hot encode the index
        return None, idx

    def collate_fn(self, batch):
        # Collate function to gather one-hot vectors and extract the submatrix of D
        _, indices = zip(*batch)

        # Convert indices to tensor for advanced indexing
        indices = torch.tensor(indices)

        # Extract the submatrix of D for these indices
        submatrix = self.distance_matrix[indices][:, indices]

        return indices, submatrix

class SammonLoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        notnull = target!=0
        x = input[notnull]/torch.sqrt(target[notnull])
        y = torch.sqrt(target[notnull])
        return nn.functional.mse_loss(x, y, reduction=self.reduction)

def sgd_mds(D,initialData,n_epochs = 1000, lr=1e-2, batch_size = None, max_epochs_no_improvement = 100, loss = 'MSE', saveloss=False):

    N = D.shape[0]
    if batch_size == None:
        batch_size = round(N/10)
    
    dataDtype = initialData.dtype
    if dataDtype!=np.float32:
        initialData = initialData.astype('float32')
    init = torch.from_numpy(initialData).type(torch.float32)
    
    D = torch.tensor(D).float()
    X = torch.nn.Parameter(init)
    optimizer = optim.Adam([X], lr=lr)

    ds = IndexDataset(D)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    if loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss =='Sammon':
        loss_fn = SammonLoss()

    best_total_loss = float('inf')
    epochs_no_improve = 0

    if saveloss:
        # Initialize lists to store losses for each epoch
        total_losses = []

    for epoch in tqdm(range(n_epochs)): # tqdm gives a progress bar in the terminal
        total_loss = 0
        for indices, distance_submatrices in dataloader:

            optimizer.zero_grad()
            out_coords = X[indices]
            out_distances = torch.cdist(out_coords, out_coords)

            loss = loss_fn(out_distances, distance_submatrices)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if saveloss:
            total_losses.append(total_loss)

        if total_loss < best_total_loss:
            best_total_loss = total_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve ==max_epochs_no_improvement:
            print(f'Convergence. Early stopping at epoch {epoch}!')
            break
    
    if saveloss:
        saveTotalLossPlots(total_losses,N)
        print('A plot of the loss over epochs was stored in ./Results/Losses/N_'+str(N)+'_total_loss.png\n')

    return X.detach().numpy().astype(dataDtype)


