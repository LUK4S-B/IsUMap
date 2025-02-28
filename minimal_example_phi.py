
from time import time
from isumap import isumap
from data_and_plots import plot_data, createMammoth, load_MNIST, printtime, createNonUniformHemisphere, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST
from multiprocessing import cpu_count

k = 15
d = 2
N = 1000
normalize = True
metricMDS = True
distBeyondNN = False
tconorm = "probabilistic sum"

if __name__ == '__main__':
    data, labels = createNonUniformHemisphere(N)
    
    
    plot_data(data,labels,title="Initial dataset",display=True, save=False)
    

    for phi in ['exp','half_normal','log_normal','pareto','uniform']:
        t0 = time()

        finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, d,
            normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None, sgd_max_epochs_no_improvement = 75, sgd_loss = 'MSE', sgd_saveloss=True, tconorm = tconorm, phi=phi)
        t1 = time()
        
        title = "Non-uniform Hemisphere N_" + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm + "phi_" + phi
        
        plot_data(finalInitEmbedding,labels,title=title,display=True, save=False)
        plot_data(finalEmbedding,labels,title=title,display=True, save=False)

        print("\nResult saved in './Results/" + title + ".png'")
        
        printtime("Isumap total time",t1-t0)
    
# possible sets in order of importance: torus, swiss-role-with-hole, möbius, s-shape, breastcancer, heart disease