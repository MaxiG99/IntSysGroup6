import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from scikeras.wrappers import KerasClassifier, KerasRegressor
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

#%%
path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data = pd.read_csv(path+'Project/data/robot_inverse_kinematics_dataset.csv')
X = data.iloc[:,6:].to_numpy()
y = data.iloc[:,0:6].to_numpy()
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_Y.fit_transform(y)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)
#%%
# Define a function to create the Keras model
# Define a function to create the Keras model
def create_model(neurons):
    model = Sequential()
    input_dim = 3
    output_dim = 6
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    #for _ in range(hidden_layers):
    model.add(Dense(neurons, activation='relu'))
        #if dropout_rate > 0.0:
        #    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model
#%%
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, verbose=0,neurons=32)
# Define hyperparameters to search

hidden_layers = [2, 3, 4]
neurons = [32,64]
dropout_rate = [0.0,0.1]

param_grid = dict(neurons=neurons)
                  #, hidden_layers=hidden_layers, dropout_rate=dropout_rate)

#%%
# Create a GridSearchCV instance
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

# Fit the grid search to your data
grid_result = grid.fit(X_train, y_train)
#%% md

#%%
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=param_grid, scoring="accuracy")