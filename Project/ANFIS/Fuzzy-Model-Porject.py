# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:53:55 2023

@author: noahe
"""

import pandas as pd
import matplotlib.pyplot as plt
from pyfume import *
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, SimpleRNN
#from tensorflow.keras.optimizers import Adam

import skfuzzy as fuzz
import sklearn as learn
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

#load the robot kinematics data
data = np.loadtxt("./Data/robot_inverse_kinematics_dataset.csv", skiprows=1, delimiter=',')

#defining the number of clusters:
nr_clusters = 3

print(data.shape)
print(type(data))

#define a scalere: here Min-Max-Scaler
scaler = learn.preprocessing.MinMaxScaler()


#Define the input and output variables
xyz_data=data[:, 6:] #end-effector kartesian (xyz) position
q_data = data[:, :6] # Robot joint angles

#Using the Min-Max-Scaler to scale the q and xyz Data
xyz_sc=scaler.fit_transform(xyz_data)
q_sc=scaler.fit_transform(q_data)
print(np.ndim(xyz_sc))
print(np.ndim(q_sc))

#split data into training data and validation data. The training Data will be 75% of the data and the test data is 25% of the whole data
#the random state is set to 42 becuase it is popular. It just allows a reproduction of the values generated, so that the results will be the same every time the script is excecuted
q_train, q_val, xyz_train, xyz_val = train_test_split(q_sc, xyz_sc, test_size=0.25, random_state=42)

#cl = Clusterer(nr_clusters, x_train=xyz_data, y_train=q_data)
#cluster_centers , partition_matrix, _ = cl.cluster(method="fcm")

all_data = np.zeros((11250,9))

all_data[:3][:]=xyz_train
all_data[3:][:]=q_train

cl = fuzz.cmeans([xyz_train,q_train], 10, 2, error=0.005, maxiter=1000)


      



