# A multilayer perceptron for border irregularity detection

import numpy as np
import os
import cv2
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import time

start = time.time()

errors = []
errors_test = []
lines = []
predictions = []
probabilities = []
labels = []
correct = 0
alphas = [0.001]

border_irregularity_file = pd.read_csv('training.csv')

y = border_irregularity_file[['label']]
X = border_irregularity_file[['FD','convexity']]

X = X.astype(np.float32).values
y = y.astype(np.float32).values

for i in y:
	labels.append(i[0])

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative (slope)
def sigmoid_output_to_derivative(output):
    return output*(1-output)

for alpha in alphas:
    np.random.seed(1)

    synapse_0 = 2*np.random.random((2,4)) - 1
    synapse_1 = 2*np.random.random((4,2)) - 1
    synapse_2 = 2*np.random.random((2,1)) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    prev_synapse_2_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    synapse_2_direction_count = np.zeros_like(synapse_2)
        
    for j in range(1):

        # Feed forward
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        layer_3 = sigmoid(np.dot(layer_2,synapse_2))

        layer_3_error = y - layer_3

        error = np.mean(np.abs(layer_3_error))
        errors.append(error)

        layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)

        layer_2_error = layer_3_delta.dot(synapse_2.T)

        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_2_weight_update = (layer_2.T.dot(layer_3_delta))
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0)) 
            synapse_2_direction_count += np.abs(((synapse_2_weight_update > 0)+0) - ((prev_synapse_2_weight_update > 0) + 0))               
        
        synapse_2 += alpha * synapse_2_weight_update
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
        prev_synapse_2_weight_update = synapse_2_weight_update

    end = time.time()
    time_elapsed = end - start

    print('training time: ')
    print(time_elapsed)

start = time.time()

border_irregularity_file_test = pd.read_csv('testing.csv')

y_test = border_irregularity_file_test[['label']]
X_test = border_irregularity_file_test[['FD','convexity']]

X_test = X_test.astype(np.float32).values
y_test = y_test.astype(np.float32).values

i = 0
for i in range(len(X_test)):
    layer_0_test = X_test[i]
    layer_1_test = sigmoid(np.dot(layer_0_test,synapse_0))
    layer_2_test = sigmoid(np.dot(layer_1_test,synapse_1))
    layer_3_test = sigmoid(np.dot(layer_2_test,synapse_2))

    probabilities.append(layer_3_test[0])

print(probabilities)

threshold = np.mean(probabilities)

threshold_index = 0
for threshold_index in range(len(probabilities)):
    if probabilities[threshold_index] > threshold:
        prediction = 1
    elif probabilities[threshold_index] <= threshold:
        prediction = 0

    print (prediction)
    predictions.append(prediction)
    
    if prediction == y_test[threshold_index][0]:
        correct = correct + 1
    
    threshold_index = threshold_index + 1

print (correct)

y_true = labels
y_probas = probabilities
metrics.plot_roc_curve(y_true, y_probas)
plt.show()

end = time.time()
time_elapsed = end - start

print ('testing time: ')
print (time_elapsed)
