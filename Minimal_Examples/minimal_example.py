
from time import time
from isumap import isumap
from data_and_plots import plot_data, createMammoth, load_MNIST, printtime, createNonUniformHemisphere, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count

k = 20
d = 2
N = 5000
normalize = True
metricMDS = True
distBeyondNN = False
tconorm = "probabilistic sum"
distFun = "canonical"
phi = "exp"
epm = False
mds_loss_fun = "ratio"
apply_Dijkstra = True
extractSubgraphs = True
max_param = 100.0

if __name__ == '__main__':
    title = "MNIST 4 N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm + " distFun_" + distFun + " phi_" + phi + " epm_" + str(epm) + " mdsLoss_" + mds_loss_fun + " apply_Dijkstra_" + str(apply_Dijkstra) + " extractSubgraphs_" + str(extractSubgraphs) + " max_param_" + str(max_param)

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
        normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, saveDistMatrix = False, initialization="cMDS", metricMDS=metricMDS, sgd_loss = mds_loss_fun, sgd_saveloss=True, tconorm = tconorm, epm=epm, distFun=distFun, phi=phi, apply_Dijkstra=apply_Dijkstra, extractSubgraphs=extractSubgraphs, max_param=max_param)
    t1 = time()
    
    plot_data(finalInitEmbedding,labels,title=title+" init",display=True, save=True, colorbar=True)
    plot_data(finalEmbedding,labels,title=title,display=True, save=True, colorbar=True)
    print("\nResult saved in './Results/" + title + ".png'")
    
    printtime("Isumap total time",t1-t0)
    
