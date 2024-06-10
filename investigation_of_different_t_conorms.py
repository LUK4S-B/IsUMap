
from time import time
from isumap import isumap
from data_and_plots import plot_data, createMammoth, load_MNIST, printtime, createNonUniformHemisphere, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count

k = 15
d = 2
N = 7000
normalize = True
metricMDS = True
distBeyondNN = True

data, labels = createNonUniformHemisphere(N)
# data, labels = createSwissRole(N,hole=True,seed=0)
# data, labels = createFourGaussians(8.2,N)
# data, labels = createMoons(numberOfPoints,noise=0.1,seed=42)
# data, labels = createTorus(N,seed=0)
# data, labels = createMammoth(N,k=30,seed=42)

# data, labels = load_MNIST(N)
# data, labels = createBreastCancerDataset()
# data, labels = load_FashionMNIST(N)

plot_data(data,labels,title="Initial dataset - Hemisphere - N_" + str(N),display=True)

for tco in ["drastic sum","Einstein sum","nilpotent maximum"]:
    print("\n\n tconorm = " + tco + "\n")
    t0=time()
    finalEmbedding, clusterLabels = isumap(data.copy(), k, d, 
        normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, labels=labels, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 3000, sgd_lr=1e-2, sgd_batch_size = None,sgd_max_epochs_no_improvement = 75, sgd_loss = 'MSE', sgd_saveplots_of_initializations=True, sgd_saveloss=True, tconorm = tco)
    t1 = time()

    title = "Non-uniform Hemisphere N_" + str(N) + " tconorm_" + tco + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS)

    plot_data(finalEmbedding,labels,title=title,display=False)

    print("\nResult saved in './Results/" + title + ".png'")
    printtime("Isumap total time",t1-t0)

