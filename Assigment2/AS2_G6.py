# %% Imports
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# %% Load dataset and create train-test sets
data = load_wine()
X = data.data
y = data.target
scaler = MinMaxScaler()
var_names = data.feature_names
var_names = [var_names[i].title().replace('/','') for i in range(0, len(var_names))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(1000,1000,1000),random_state=42, max_iter=500)
regr.fit(X_train, y_train)

# %% Get model predictions
y_pred = regr.predict(X_test)

# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

np.savetxt('y_pred_Group6_ass2_mlp.txt',y_pred,delimiter=',',fmt='%.0f')