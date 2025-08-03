import pickle
import matplotlib.pyplot as plt

import os, sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../results/Classifier_Experiments/")

with open(scriptPath + '/../results/Classifier_Experiments/results_cifar_no_normalization.pkl', 'rb') as f:
    rs = pickle.load(f)

rss = rs['classifier_results']

uma_test_accs = rss['umap']['test_accs']
isu_test_accs = rss['isumap']['test_accs']
iso_test_accs = rss['isomap']['test_accs']

ds = rss['isomap']['ds']

fig = plt.figure(figsize=(12, 12))

plt.plot(ds,uma_test_accs,label="UMAP",c="orange")
plt.scatter(ds,uma_test_accs,s=25,marker='o',c="orange")

plt.plot(ds,isu_test_accs,label="IsUMap",c="blue")
plt.scatter(ds,isu_test_accs,s=25,marker='^',c="blue")

plt.plot(ds,iso_test_accs,label="Isomap",c="green")
plt.scatter(ds,iso_test_accs,s=25,marker='s',c="green")


plt.title("CIFAR-10 classifier test accuracies")
plt.xscale('log',base=2)
plt.xticks(ds,[str(d) for d in ds])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("Embedding dimension")
plt.ylabel("Classifier test accuracy")

# Create legend
from matplotlib.lines import Line2D
custom_legend = [Line2D([0], [0], color='orange', linestyle='-', lw=2, marker='o', markersize=5, label='UMAP'),Line2D([0], [0], color='blue', linestyle='-', lw=2, marker='^', markersize=5, label='IsUMap'),Line2D([0], [0], color='green', linestyle='-', lw=2, marker='s', markersize=5, label='Isomap')]
plt.legend(handles=custom_legend, handlelength=4)

plt.show()
