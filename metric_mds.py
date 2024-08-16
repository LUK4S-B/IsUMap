import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data_and_plots import saveTotalLossPlots

class IndexDataset(Dataset):
    ''' A Dataset class to return submatrices of distance matrices together with the corresponding indices.'''
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
    r'''
    Loss class implementing the sammon loss for multdimensional scaling.
    The sammon loss of a collection of target distances $d_{ij}$ and approximate distances $\rho_{ij}$ is given by
    $$L = \sum_{ij} \vert \vert \frac{\rho_{ij}}{\sqrt{d_{ij}}} - d_{ij} \vert \vert$$

    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        notnull = target!=0
        x = input[notnull]/torch.sqrt(target[notnull])
        y = torch.sqrt(target[notnull])
        return nn.functional.mse_loss(x, y, reduction=self.reduction)

def sgd_mds(D,initialData,
            n_epochs:int = 1000,
            lr:float=1e-2,
            batch_size = None,
            max_epochs_no_improvement = 100,
            loss = 'MSE',
            saveloss:bool=False):

    r'''

    Performs (metric) multidimensional scaling via a stochastic gradient descent (SGD).
    Given a set of target distances $D \in \mathbb{R}^{n \times n}$, points in a lower dimensional space $y_1,...,y_n$
    are optimized via SGD by means of minimizing a loss function $L(D,D_Y)$ where $D_Y$ is the distance matrix
    of the points $y_i$.

    :param D: np.ndarray or torch.tensor shape: (n,n)
    :param initialData: np.ndarray or torch.tensor, shape: (n,d)
    :param n_epochs: int, number of epochs
    :param lr: float, learning rate
    :param batch_size: int or None, number of samples per batch, if None then default value of D.shape[0]/10
    :param max_epochs_no_improvement: int, max number of epochs without improvement that triggers early stopping
    :param loss: str or torch.nn.modules.loss._Loss, loss function, implemented are 'MSE' and 'Sammon'
    :param saveloss: bool, whether to save loss progress
    :return: X: np.ndarray (n,d) - low dimensional embedding
    '''


    N = D.shape[0]
    if batch_size == None:
        batch_size = round(N/10)
        if batch_size == 0:
            if N>1:
                print("Directly returning initial embedding because there are only " + str(N) + " points in the cluster")
            else:
                print("Directly returning initial embedding because there is only " + str(N) + " point in the cluster")
            return initialData
    
    dataDtype = initialData.dtype
    init = convert_to_torch_float32(initialData)
    D = convert_to_torch_float32(D)


    X = torch.nn.Parameter(init)
    optimizer = optim.Adam([X], lr=lr)

    ds = IndexDataset(D)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    if loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss =='Sammon':
        loss_fn = SammonLoss()
    elif isinstance(loss,nn.modules.loss._Loss):
        loss_fn = loss
    else:
        raise NotImplementedError


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
        saveTotalLossPlots(total_losses,N,'Metric MDS - Loss per epoch')
        print('A plot of the loss over epochs was stored in ./Loss_graphs/N_'+str(N)+'_total_loss.png\n')

    return X.detach().numpy().astype(dataDtype)


def convert_to_torch_float32(initialData):
    # Check if the input data is a NumPy array
    if isinstance(initialData, np.ndarray):
        # Ensure the data type is float32
        if initialData.dtype != np.float32:
            initialData = initialData.astype('float32')
        # Convert the NumPy array to a PyTorch tensor
        init = torch.from_numpy(initialData)
    # Check if the input data is a PyTorch tensor
    elif isinstance(initialData, torch.Tensor):
        # Ensure the data type is float32
        if initialData.dtype != torch.float32:
            initialData = initialData.to(dtype=torch.float32)
        init = initialData
    else:
        raise TypeError('Initial data must be either a NumPy array or a PyTorch tensor.')

    # Ensure the tensor type is torch.float32 (redundant but explicitly safe)
    init = init.type(torch.float32)
    return init