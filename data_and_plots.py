import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from sklearn import datasets
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import pickle
import os
import pickle
import plotly.express as px
import torchvision

def plotly_data(data,labels,title):
    fig = px.scatter(x=data[:,0],y=data[:,1],color=labels); 
    fig.layout.yaxis.scaleanchor="x"
    fig.layout.paper_bgcolor='rgba(0,0,0,0)'
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.write_html("./Results/"+title+".html")

def createFourGaussians(distance,numberOfPoints):
    def generate_points(means,covs,ns):
        assert len(means)==len(covs)==len(ns)
        
        X= [np.random.multivariate_normal(means[i],covs[i],size=(ns[i],)) for i in range(len(ns))]
        X= np.vstack(X)
        
        y = [np.ones(ns[i])*i for i in range(len(ns))]
        y = np.hstack(y)
        
        return X,y

    pointsPerGaussian = round(numberOfPoints/4)
    means = [np.array([-4*distance,0]),np.array([1.2*distance,0]),np.array([0,1*distance]),np.array([0,-2*distance])]
    covs = [2*np.eye(2),2*np.eye(2),2*np.eye(2),2*np.eye(2)]
    ns = [pointsPerGaussian,pointsPerGaussian,pointsPerGaussian,pointsPerGaussian]

    dataset,colors = generate_points(means,covs,ns)
    return dataset,colors

def createMoons(numberOfPoints,noise=0.1,seed=42):
    ms = datasets.make_moons(n_samples=numberOfPoints,noise=noise,random_state=np.random.RandomState(seed))
    dataset = ms[0]
    colors = ms[1]
    return dataset,colors

def createNonUniformHemisphere(N,seed0=0,k=30):
    rng = np.random.default_rng(seed=seed0)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    phi = rng.uniform(0, 0.5 * np.pi, size=N)
    t = rng.uniform(0,1,size = N)
    P=  np.arccos(1 - 2 * t)

    if ((P < (np.pi - (np.pi / 4))) & (P > ((np.pi / 4)))).all():
        indices = (P < (np.pi - (np.pi / 4))) & (P > ((np.pi /4)))
        x,y,z = (np.sin(P[indices]) * np.cos(theta[indices]),
                np.sin(P[indices]) * np.sin(theta[indices]),
                np.cos(P[indices]))            
    else :
        x,y,z =(np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi))
    
    data = np.array([x,y,z]).T

    # for coloring the hemisphere:
    M_train, M_test, N_train, N_test = train_test_split(data, data, test_size=int(N/2), random_state=seed0)
    AC = AgglomerativeClustering(n_clusters=7, linkage='ward')
    AC.fit(M_train)
    labels = AC.labels_
    KN = KNeighborsClassifier(n_neighbors=k)
    KNN = KN.fit(M_train,labels)
    colors = KN.predict(data)
    
    return data, colors

def createTorus(N,seed=0):
    random_state = check_random_state(seed)
    theta = random_state.rand(N) * (2 * np.pi)
    phi = random_state.rand(N) * (2 * np.pi)
    colors = phi
    dataset = np.array([(2+np.cos(theta))*np.cos(phi),
            (2+np.cos(theta))*np.sin(phi),
            np.sin(theta)]).T
    return dataset,colors

def createSwissRole(N,hole=True,seed=0):
    return datasets.make_swiss_roll(n_samples=N, hole = hole, noise=0.0 , random_state=seed)

def createBreastCancerDataset():
    data = pd.read_csv('Dataset_files/BreastCancerDataset.csv')
    data = data.drop('id',axis=1)
    data=data.drop('Unnamed: 32',axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    X = data.values
    colors=data['diagnosis']
    return X,colors

def createMammoth(N,k=30,seed=42):
    mammoth = pd.read_csv('Dataset_files/mammoth.csv')
    if N <= len(mammoth):
        mammoth_sample = mammoth.sample(N)
    else:
        mammoth_sample = mammoth
    
    # for coloring the mammuth:
    M_train, M_test, N_train, N_test = train_test_split(mammoth_sample, mammoth_sample, test_size=int(N/2), random_state=seed)
    AC = AgglomerativeClustering(n_clusters=11, linkage='ward')
    AC.fit(M_train)
    labels = AC.labels_
    KN = KNeighborsClassifier(n_neighbors=k)
    KNN = KN.fit(M_train,labels)
    colors = KN.predict(mammoth_sample)

    return np.array(mammoth_sample), colors


def make_s_curve_with_hole(n_samples=4000, noise=0.05, random_state=42, hole_center=(0, 1, 0), hole_radius=0.5):
    """Generate an S curve dataset with a hole.

    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the S curve.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.

    hole_center : tuple of floats, default=(0, 1, 0)
        The center of the hole in the 3D space.

    hole_radius : float, default=0.5
        The radius of the hole.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points with a hole.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample.
    """
    generator = check_random_state(random_state)

    t = 3 * np.pi * (generator.uniform(size=(1, n_samples)) - 0.5)
    X = np.empty(shape=(n_samples, 3), dtype=np.float64)
    X[:, 0] = np.sin(t)
    X[:, 1] = 2.0 * generator.uniform(size=n_samples)
    X[:, 2] = np.sign(t) * (np.cos(t) - 1)
    X += noise * generator.standard_normal(size=(3, n_samples)).T
    t = np.squeeze(t)

    # Define the hole
    hole_center = np.array(hole_center)
    distances_squared = np.sum((X - hole_center) ** 2, axis=1)

    # Exclude points that fall within the hole
    mask = distances_squared > hole_radius ** 2
    X = X[mask]
    t = t[mask]

    return X, t


###

def load_and_store_data_file(N,filename,filetype='.pkl', normalize=True, torch_dataset=False):
    # Check if the file already exists
    if not os.path.exists('Dataset_files/'+filename+filetype):
        print("\nDownloading '"+filename+"' data.")
        os.makedirs('Dataset_files', exist_ok=True)
        if torch_dataset:
            if filename=='cifar10':
                trainset = torchvision.datasets.CIFAR10(root='./Dataset_files', train=True, download=True)
            else:
                raise("This dataset is not available")
            data = np.array(trainset.data, dtype=np.float32)
            data = data.reshape(data.shape[0],np.prod(data.shape[1:]))
            data = data / data.max()
            labels = np.array(trainset.targets)
        else:
            ddata = datasets.fetch_openml(filename)
            # Create the output directory if it doesn't exist
            data = np.array(ddata['data'])
            labels = np.array(ddata['target'])
        # Save the dataset as a .pkl file
        with open('Dataset_files/'+filename+filetype, 'wb') as f:
            pickle.dump((data, labels), f)

        print("Download successful. The files are stored in 'Dataset_files/"+filename+filetype+"' and are directly loaded from there in case you run this script a second time.")
    print("\nLoading '"+filename+"' data from file")
    with open('Dataset_files/'+filename+filetype, 'rb') as f:
        data, labels = pickle.load(f)
    print("Selecting subset of N = ",N)
    indices = random.sample(range(len(data)), N)
    data = np.array(data[indices],dtype=np.float32)
    # normalize data:
    if normalize:
        data = data / np.max(data)
    labels = np.array(labels[indices],dtype=np.int64)
    return data,labels

def load_MNIST(N):
    return load_and_store_data_file(N, 'mnist_784')

def load_FashionMNIST(N):
    return load_and_store_data_file(N, 'fashion-mnist')

def load_CIFAR_10(N):
    return load_and_store_data_file(N, 'cifar10', torch_dataset=True)

def load_heart_disease_dataset():
    heart_disease_data = pd.read_csv('./Dataset_files/heart.csv')
    X = heart_disease_data.iloc[:, :-1]
    labels = heart_disease_data.iloc[:, -1]
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    return data, labels

### Plotting data

def plot_MNIST_samples(images, labels, num_samples=10):
    num_rows = num_samples // 8 + (num_samples % 8 > 0)  # Calculate the number of rows
    plt.figure(figsize=(12, 12))
    jet = cm.get_cmap('jet', 10)
    for i in range(num_samples):
        plt.subplot(num_rows, 8, i+1)
        image = images.iloc[i].values.reshape(28, 28)  # reshape the data into a 28x28 array
        label = labels.iloc[i]  # get the label of the image
        color = jet(label / 9.)
        cmap = mcolors.ListedColormap(['white', color])  # create a custom color map
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
    plt.tight_layout()  # Adjust the padding between and around the subplots
    plt.show()

def plot_data(data,labels,title='Data',save=True,display=False,axis=False):
    if data.shape[0]==labels.shape[0]:
        fig = plt.figure(figsize=(12, 12))
        plt.title(title)
        dim = data.shape[1]
        if dim==2:
            plt.scatter(data[:,0],data[:,1],s=3,c=labels, cmap="jet")
            # plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='datalim')
            if not axis:
                plt.axis('off')
        elif dim==3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(data[:,0],data[:,1],data[:,2],s=3,c=labels, cmap="jet")
            ax.view_init(20, -20)
        else:
            print("Invalid dimension for plot")
        if save and (dim==2 or dim==3):
            # Create the output directory if it doesn't exist
            os.makedirs('Results', exist_ok=True)
            plt.savefig('./Results/'+title+'.png')
        if display:
            plt.show()
        else:
            plt.close()

def saveTotalLossPlots(total_losses,N,title='Loss per epoch'):
    # Plot total loss
    plt.figure()
    plt.plot((total_losses))
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    loss_graph_path = './Loss_graphs/'
    if not os.path.exists(loss_graph_path):
        os.makedirs(loss_graph_path)
    plt.savefig(loss_graph_path+str(N)+title+'.png')
    plt.close()


### other helper functions

def printtime(name,delta_t):
    if delta_t > 120:
        print("\n"+name+": %.2f min" % (delta_t / 60))
    else:
        print("\n"+name+": %.2f sec" % delta_t)