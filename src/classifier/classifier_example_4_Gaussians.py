from Module_classifier import SimpleLinearClassifier, Classifier, train, plot_decision_boundary

import torch
import os
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../")
from data_and_plots import createFourGaussians, plot_data

k = 15
d = 2
N = 300
normalize = False
metricMDS = True
distBeyondNN = False
tconorm = "canonical"

data, labels = createFourGaussians(8.2,N)
plot_data(data,labels,title='Data with native labels',save=False,display=True,axis=True)

input_dim = data.shape[1]
output_dim = 4
# hidden_layer_dim = 4
# Cnet = Classifier(input_dim,output_dim,hidden_layer_dim)
Cnet = SimpleLinearClassifier(input_dim,output_dim)

train(Cnet,data,labels)

class_probabilities = Cnet(torch.tensor(data,dtype=torch.float32))
c_labels = torch.argmax(class_probabilities,dim=1)
plot_data(data,c_labels,title='Data with classifier labels',save=False,display=True)

plot_decision_boundary(Cnet, -40.0, 15.0, -30.0, 20.0, h=0.01, data=data, labels=labels)