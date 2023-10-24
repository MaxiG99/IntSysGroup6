#sequence q3,4,2,5,1
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score
import math
# %%
path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data = pd.read_csv(path+'Project/data/robot_inverse_kinematics_dataset.csv')
q3_pred= pd.read_csv(path+"Project/saved_pred/q3_pred.csv")
#data = pd.concat([data,q3_pred],axis=1)
X = data.iloc[:, 6:]
y = data.iloc[:, 0:5]

y = y.drop(['q3',],axis=1) #removing column 3 '''

X = pd.concat([X,q3_pred],axis=1)
scaler_X = MinMaxScaler()
scalers = [MinMaxScaler() for _ in range(5)]

for i,column in enumerate(y.columns):
    y[[column]] = scalers[i].fit_transform(y[[column]])

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.2)

# %%
def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dense(1)
    ])
    return model


#%%
histories = []
mae_scores = []
r2_scores = []

for i, column in enumerate(y.columns):
    print(f"Training model for {column}")
    model = create_model()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(learning_rate=0.001, weight_decay=0.01)
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train[column],
        validation_split=0.2,
        verbose=1,
        epochs=50,
        callbacks=[early_stopping]
    )
    histories.append(history)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test[column], y_pred)

    
    # Inverse scaling to get MAE in degrees
    y_pred_inverse = scalers[i].inverse_transform(y_pred)
    y_test_inverse = scalers[i].inverse_transform(y_test[[column]])
    mae_degrees = mean_absolute_error(y_test_inverse, y_pred_inverse)
    
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    mae_scores.append(mae_degrees)
    r2_scores.append(r2)

    print(f"MAE for {column}: {mae_degrees} degrees")
    print(f"R-squared (R2) Score for {column}: {r2}")
#%%

for i, column in enumerate(y.columns):
    print(f"MAE for {column}: {math.degrees(mae_scores[i])} degrees")
    print(f"R-squared (R2) Score for {column}: {r2_scores[i]*100}")

#%%

# Plot the loss curves for all models
import matplotlib.pyplot as plt

for i, history in enumerate(histories):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves for {y.columns[i]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

plt.show()


#%%
initial_learning_rate = 0.001
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,  # Adjust this value
    decay_rate=0.5,  # Adjust this value
    staircase=True
)
callbacks.LearningRateScheduler(lr_schedule)


model.compile(loss='mean_squared_error',
              optimizer=optimizers.Adam(learning_rate=lr_schedule,
                                        weight_decay=0.01))



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
          epochs=50,
          callbacks=[early_stopping])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# %%
y_pred = model.predict(X_test)
print("MAE Absolute: " ,mean_absolute_error(y_test, y_pred))
y_pred = scaler_Y.inverse_transform(y_pred)
y_test = scaler_Y.inverse_transform(y_test)
print("MAE in deg ",mean_absolute_error(np.degrees(y_test), np.degrees(y_pred)))
print("MAE in deg ",math.degrees(mean_absolute_error(y_test, y_pred)))

r2 = r2_score(y_test, y_pred)
print("R-squared (R2) Score:", r2*100)
plt.show()
# %%
# Save Model
import joblib

model_filename = "Project/Models/MLP_q3.pkl"
path= '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

joblib.dump(model, path+model_filename)
# %%

full_prediction_q3 = model.predict(X)
full_prediction_q3 = scaler_Y.inverse_transform(full_prediction_q3)
results_qf = pd.DataFrame({'q3_pred': full_prediction_q3.flatten()})

# Save the predictions to a CSV file
results_filename = "Project/saved_pred/q3_pred.csv"

results_qf.to_csv(path+results_filename, index=False)
