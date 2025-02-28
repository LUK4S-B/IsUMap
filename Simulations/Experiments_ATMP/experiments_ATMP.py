# see: https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
# cells: bone marrow mononuclear cells of healthy human donors
import os
path = os.path.dirname(__file__)
from pathlib import Path
path = Path(path).parent.absolute().parent.absolute()
path = str(path)
import sys
sys.path.append(path)

from time import time
from isumap import isumap
from data_and_plots import plot_data, printtime, createNonUniformHemisphere, createMammoth, load_MNIST, createSwissRole, createFourGaussians, createMoons, createTorus, load_FashionMNIST, make_s_curve_with_hole, create_nonuniform_Mobius, createBreastCancerDataset
from multiprocessing import cpu_count
import numpy as np


k = 15
d = 2
N = 5000
metricMDS = True
normalize = True
distBeyondNN = False
tconorm = "m_scheme"
epm = False
distFun = "canonical"

tconorms = ["m_scheme", "m_scheme_Wiener_Shannon", "m_scheme_Composition", "m_scheme_Hyperbolic"]
def create_data(i):
    if i==0:
        return ("swissRole", createSwissRole(N))
        # return ("s-shape", make_s_curve_with_hole(N))
    elif i==1:
        return ("breastCancerData", createBreastCancerDataset(path="../../Dataset_files/BreastCancerDataset.csv"))
        # return ("torus", createTorus(N))
    elif i==2:
        return ("fashionMNIST", load_FashionMNIST(N))
        # return ("mobius", create_nonuniform_Mobius(N))
    elif i==3:
        return ("twoMoons", createMoons(N))
    elif i==4:
        return ("mnist", load_MNIST(N))


total_num_exps = 4*3*5 + 4

if __name__ == '__main__':
    counter = 0
    for i in range(5):

        dataset_name, (data, labels) = create_data(i)

        for tconorm in tconorms:

            if tconorm == "m_scheme_Hyperbolic":
                m_values = [1.0]
            else:
                m_values = [0.01, 0.25, 0.5, 0.75, 1.0]

            for m_scheme_value in m_values:

                title = dataset_name + " " + str(N) + " k_" + str(k) + " beyondNN_" + str(distBeyondNN) + " normalize_" + str(normalize) + " metricMDS_" + str(metricMDS) + " tconorm_" + tconorm + " m_value_" + str(m_scheme_value) + " epm_" + str(epm) + " distFun_" + distFun

                counter += 1
                print("\n\n")
                print("counter", counter, " / ", total_num_exps)
                image_path = os.path.join('./Results/', title + ".png")
                if os.path.isfile(image_path):
                    print("Image already exists. Skipping.")
                    continue

                print("dataset_name", dataset_name)
                print("tconorm", tconorm)
                print("m_scheme_value", m_scheme_value)

                t0=time()
                finalInitEmbedding, finalEmbedding, clusterLabels = isumap(data, k, d,
                    normalize = normalize, distBeyondNN=distBeyondNN, verbose=True, dataIsDistMatrix=False, dataIsGeodesicDistMatrix = False, saveDistMatrix = False, initialization="cMDS", metricMDS=metricMDS, sgd_n_epochs = 1500, sgd_lr=1e-2, sgd_batch_size = None, sgd_max_epochs_no_improvement = 75, sgd_loss = 'MSE', sgd_saveloss=True, tconorm = tconorm, epm=epm, m_scheme_value=m_scheme_value, distFun=distFun)
                t1 = time()
                printtime("Isumap total time",t1-t0)

                plot_data(finalInitEmbedding,labels,title=title + " init",display=False, save=True)
                plot_data(finalEmbedding,labels,title=title,display=False, save=True)
