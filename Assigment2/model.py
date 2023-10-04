import joblib
from sklearn.neural_network import MLPClassifier
import numpy as np

model_filename = 'mlp_model.pkl'
loaded_model = joblib.load(model_filename)

new_data = [] # Add real Data of the wine features 
predictions = loaded_model.predict(new_data)

# in case you want to safe the predictions uncomment next line
#np.savetxt('Predictions_Group6_ass2_mlp.txt',y_pred,delimiter=',',fmt='%.0f')