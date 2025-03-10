
from time import time
from isumap import isumap
from data_and_plots import plot_data, createMammoth, load_MNIST, printtime, createNonUniformHemisphere, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count

k = 15
d = 2
N = 5000

epm = False
normalize = True
distBeyondNN = False
tconorm = "probabilistic sum"
distFun = "canonical"

if __name__ == '__main__':
    title = "MNIST N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " tconorm_" + tconorm + " distFun_" + distFun + " phi_exp" + " epm_" + str(epm)

    # data, labels = createNonUniformHemisphere(N)
    # data, labels = createSwissRole(N,hole=True,seed=0)
    # data, labels = createFourGaussians(8.2,N)
    # data, labels = createMoons(numberOfPoints,noise=0.1,seed=42)
    # data, labels = createTorus(N,seed=0)
    # data, labels = createMammoth(N,k=30,seed=42)
    data, labels = load_MNIST(N)
    # data, labels = createBreastCancerDataset()
    # data, labels = load_FashionMNIST(N)

    # plot_data(data,labels,title="Initial dataset",display=True, save=False)
    
    t0=time()
    finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, d,
        normalize = normalize, distBeyondNN=distBeyondNN, sgd_saveloss=True, tconorm = tconorm, epm=epm, distFun=distFun)
    t1 = time()
    
    plot_data(finalInitEmbedding,labels,title=title+" init",display=True, save=True)
    plot_data(finalEmbedding,labels,title=title,display=True, save=True)
    print("\nResult saved in './Results/" + title + ".png'")
    
    printtime("Isumap total time",t1-t0)
    
