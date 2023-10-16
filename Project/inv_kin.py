# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
import math
# %%
path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data = pd.read_csv(path+'Project/data/robot_inverse_kinematics_dataset.csv')
X = data.iloc[:,6:].to_numpy()
y = data.iloc[:,2:3].to_numpy()
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_Y.fit_transform(y)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)

# %%
model = tf.keras.Sequential([
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dense(1)
])


model.compile(loss='mean_squared_error',
              optimizer=optimizers.Adam(0.0001))



# %%

early_stopping = callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
          X_train,
          y_train,
          validation_split=0.2,
          verbose=1,
          epochs=100,
          callbacks=[early_stopping])

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# %%
y_pred = model.predict(X_test)
print("MAE Absolute: " ,mean_absolute_error(y_test, y_pred))
y_pred = scaler_Y.inverse_transform(y_pred)
y_test = scaler_Y.inverse_transform(y_test)
print("MAE in deg ",math.degrees(mean_absolute_error(y_test, y_pred)))

ranges_joints = [330, 220,180, 320,240,360]

plt.show()
# %%
# Save Model
import joblib

model_filename = "Project/Models/MLP_q3.pkl"
path= '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

joblib.dump(model, path+model_filename)
# %%

full_prediction_q3 = model.predict(X)
results_qf = pd.DataFrame({'q3_pred': full_prediction_q3.flatten()})

# Save the predictions to a CSV file
results_filename = "Project/saved_pred/q3_pred.csv"

results_qf.to_csv(path+results_filename, index=False)
# %%
