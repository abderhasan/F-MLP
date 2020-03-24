# Type-II fuzzy multilayer perceptron for skin lesion border irregularity detection

import numpy as np
import pandas as pd
from fuzzy_c import fuzzy, import_data
from sympy import symbols, diff, simplify
import scikitplot as skplt
import matplotlib.pyplot as plt
import math
import os
import cv2
import time

start = time.time()

alphas = [0.001]
errors_upper = []
errors_lower = []

lower_probabilities = []
upper_probabilities = []
final_probabilities = []

upper_predictions = []
final_predictions = []

correct = 0

border_irregularity_file = pd.read_csv('training.csv')

y = border_irregularity_file[['label']]
X = border_irregularity_file[['FD','convexity']]

X = X.astype(np.float32).values
y = y.astype(np.float32).values

X_to_text_file = np.savetxt('x.txt', X.reshape(np.shape(X)), fmt='%5f')
X_data = import_data('x.txt')
X_membership_matrix = fuzzy(X_data,2,2)
X_degree_of_membership_first_class = np.array(X_membership_matrix)[:,0]
X_degree_of_membership_second_class = np.array(X_membership_matrix)[:,1]
X_degree_of_membership_difference = abs(np.array(X_membership_matrix)[:,0] - np.array(X_membership_matrix)[:,1]) ** 2
X_degree_of_membership_difference = np.mean(X_degree_of_membership_difference)

old_settings = np.seterr(all='ignore')

def sigmoid_upper(x):
    output = 1/(1+np.exp(-x))
    output = output ** 2
    return output

def sigmoid_lower(x):
    output = 1/(1+np.exp(-x))
    output = (output) ** 0.5
    return output

def sigmoid_output_to_derivative(output):
    return output * (1 - output)
    
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
    
    for j in xrange(1):
        layer_0 = X
        layer_1 = sigmoid_upper(np.dot(layer_0,synapse_0))
        
        layer_1_to_text_file = np.savetxt('layer1_upper.txt', layer_1.reshape(np.shape(layer_1)), fmt='%5f')
        layer_1_data = import_data('layer1_upper.txt')
        layer_1_membership_matrix = fuzzy(layer_1_data,2,2)
        layer_1_degree_of_membership_second_class = np.array(layer_1_membership_matrix)[:,1]
        layer_1_degree_of_membership_difference = abs(np.array(layer_1_membership_matrix)[:,0] - np.array(layer_1_membership_matrix)[:,1]) ** 2
        layer_1_degree_of_membership_difference = np.mean(layer_1_degree_of_membership_difference)

        layer_2 = sigmoid_upper(np.dot(layer_1,synapse_1))
        layer_2_to_text_file = np.savetxt('layer2_upper.txt', layer_2.reshape(np.shape(layer_2)), fmt='%5f')
        layer_2_data = import_data('layer2_upper.txt')
        layer_2_membership_matrix = fuzzy(layer_2_data,2,2)
        layer_2_degree_of_membership_second_class = np.array(layer_2_membership_matrix)[:,1]
        layer_2_degree_of_membership_difference = abs(np.array(layer_2_membership_matrix)[:,0] - np.array(layer_2_membership_matrix)[:,1]) ** 2
        layer_2_degree_of_membership_difference = np.mean(layer_2_degree_of_membership_difference)

        layer_3 = sigmoid_upper(np.dot(layer_2,synapse_2))

        layer_3_error = y - layer_3

        error_upper = np.mean(np.abs(layer_3_error))
        errors_upper.append(error_upper)

        layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)

        layer_2_error = layer_3_delta.dot(synapse_2.T)

        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_2_weight_update = (layer_2.T.dot(layer_3_delta))
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        synapse_2_upper = synapse_2 + layer_2_degree_of_membership_difference  * alpha * synapse_2_weight_update
        synapse_1_upper = synapse_1 + layer_1_degree_of_membership_difference  * alpha * synapse_1_weight_update
        synapse_0_upper = synapse_0 + X_degree_of_membership_difference * alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
        prev_synapse_2_weight_update = synapse_2_weight_update

        layer_1 = sigmoid_lower(np.dot(layer_0,synapse_0))
        
        layer_1_to_text_file = np.savetxt('layer1_lower.txt', layer_1.reshape(np.shape(layer_1)), fmt='%5f')
        layer_1_data = import_data('layer1_lower.txt')
        layer_1_membership_matrix = fuzzy(layer_1_data,2,2)
        layer_1_degree_of_membership_first_class = np.array(layer_1_membership_matrix)[:,0]
        layer_1_degree_of_membership_second_class = np.array(layer_1_membership_matrix)[:,1]
        layer_1_degree_of_membership_difference = abs(np.array(layer_1_membership_matrix)[:,0] - np.array(layer_1_membership_matrix)[:,1]) ** 2
        layer_1_degree_of_membership_difference = np.mean(np.array(layer_1_membership_matrix)[:,0] - np.array(layer_1_membership_matrix)[:,1]) ** 2

        layer_2 = sigmoid_lower(np.dot(layer_1,synapse_1))
        layer_2_to_text_file = np.savetxt('layer2_lower.txt', layer_2.reshape(np.shape(layer_2)), fmt='%5f')
        layer_2_data = import_data('layer2_lower.txt')
        layer_2_membership_matrix = fuzzy(layer_2_data,2,2)
        layer_2_degree_of_membership_second_class = np.array(layer_2_membership_matrix)[:,1]
        layer_2_degree_of_membership_difference = abs(np.array(layer_2_membership_matrix)[:,0] - np.array(layer_2_membership_matrix)[:,1]) ** 2
        layer_2_degree_of_membership_difference = np.mean(layer_2_degree_of_membership_difference)

        layer_3 = sigmoid_lower(np.dot(layer_2,synapse_2))

        layer_3_error = y - layer_3

        error_lower = np.mean(np.abs(layer_3_error))
        errors_lower.append(error_lower)

        layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)

        layer_2_error = layer_3_delta.dot(synapse_2.T)

        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_2_weight_update = (layer_2.T.dot(layer_3_delta))
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        synapse_2_lower = synapse_2 + layer_2_degree_of_membership_difference  * alpha * synapse_2_weight_update
        synapse_1_lower = synapse_1 + layer_1_degree_of_membership_difference  * alpha * synapse_1_weight_update
        synapse_0_lower = synapse_0 + X_degree_of_membership_difference * alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
        prev_synapse_2_weight_update = synapse_2_weight_update

    end = time.time()

    elapsed_time = end - start
    print 'training time'
    print elapsed_time

start = time.time()

border_irregularity_file_test = pd.read_csv('testing.csv')

y_test = border_irregularity_file_test[['label']]
X_test = border_irregularity_file_test[['FD','convexity']]

X_test = X_test.astype(np.float32).values
y_test = y_test.astype(np.float32).values

correct = 0
i = 0
for i in range(len(X_test)):
    layer_0_test = X_test[i]
    layer_1_test = sigmoid_upper(np.dot(layer_0_test,synapse_0_upper))
    layer_2_test = sigmoid_upper(np.dot(layer_1_test,synapse_1_upper))
    layer_3_test = sigmoid_upper(np.dot(layer_2_test,synapse_2_upper))

    upper_probabilities.append(layer_3_test[0])

threshold_upper = np.mean(upper_probabilities)

threshold_upper_index = 0
for threshold_upper_index in range(len(upper_probabilities)):
    if upper_probabilities[threshold_upper_index] > threshold_upper:
        prediction = 1
    elif upper_probabilities[threshold_upper_index] <= threshold_upper:
        prediction = 0
    
    print prediction

    if prediction == y_test[threshold_upper_index][0]:
        correct = correct + 1

    upper_predictions.append(prediction)
    
    threshold_upper_index = threshold_upper_index + 1

print 'upper'
print correct

i = 0
for i in range(len(X_test)):
    layer_0_test = X_test[i]
    layer_1_test = sigmoid_lower(np.dot(layer_0_test,synapse_0_lower))
    layer_2_test = sigmoid_lower(np.dot(layer_1_test,synapse_1_lower))
    layer_3_test = sigmoid_lower(np.dot(layer_2_test,synapse_2_lower))

    lower_probabilities.append(layer_3_test[0])

threshold_lower = np.mean(lower_probabilities)

correct = 0
threshold_lower_index = 0
for threshold_lower_index in range(len(lower_probabilities)):
    if lower_probabilities[threshold_lower_index] > threshold_lower:
        prediction = 1
    elif lower_probabilities[threshold_lower_index] <= threshold_lower:
        prediction = 0
    
    if prediction == y_test[threshold_lower_index][0]:
        correct = correct + 1
    
    threshold_lower_index = threshold_lower_index + 1

print 'lower'
print correct

end = time.time()

elapsed_time = end - start
print 'testing time'
print elapsed_time