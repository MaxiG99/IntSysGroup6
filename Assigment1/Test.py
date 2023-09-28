# -*- coding: utf-8 -*-
#"if __name__ == '__main__':"
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from pyfume import *
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

nr_clusters=10


#data3=pyFUME(datapath="./wine/wine_data.csv",nr_clus=3, method='Takagi-Sugeno',variable_names=['Alc','Malic_Acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanth', 'Color_intensity', 'Hue', 'OD280_OD315_of', 'Proline' ] )

# Load and normalize the data (normalization by min-max normalization)
#data1 = DataLoader("./wine/wine_data.csv", variable_names=['Alc','Malic_Acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanth', 'Color_intensity', 'Hue', 'OD280_OD315_of', 'Proline' ], normalize=True )
data = load_wine()
variable_names=['Alc','Malic_Acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanth', 'Color_intensity', 'Hue', 'OD280_OD315_of', 'Proline' ]
#print(variable_names)
#data1X=data1.dataX
#data1Y=data1.dataY
data_X = data.data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_X)
dataX=normalized_data
dataY = data.target
#print(data1.dataX)
#print(data1.dataY)

# Split the data using the hold−out method in a training (default:75%)
# and test set (default:25%) .
ds = DataSplitter()
x_train, y_train , x_test, y_test = ds.holdout(dataX=dataX, dataY=dataY)

# Cluster the training data (in input−output space) using FCM with default settings
cl = Clusterer(nr_clusters,x_train, y_train)
cluster_centers , partition_matrix, _ = cl.cluster(method="fcm")

# Estimate the membership functions of the system (default:mfshape = gaussian)
ae= AntecedentEstimator(x_train, partition_matrix)
antecedent_parameters = ae.determineMF()

# Estimate the parameters of the consequent (default:global fitting=True)
ce= ConsequentEstimator(x_train, y_train, partition_matrix)
consequent_parameters = ce.suglms()

# Build a first−order Takagi−Sugeno model using Simpful
simpbuilder = SugenoFISBuilder(antecedent_parameters, consequent_parameters, variable_names)
model = simpbuilder.get_model()


# Calculate the mean squared error and the mean absolute percantage error of the model using the test data set
er = SugenoFISTester(model,  x_test , variable_names, golden_standard=y_test)

#error_mean_percent = er.calculate_MAPE()

error_mean_squared = er.calculate_MSE()

#predict the y value round it and get the absolute value so make it positive
y_pred = er.predict()

y_pred = np.round(y_pred[0])
y_pred = abs(y_pred)

#save the file as a txt file
#np.savez(file, args, kwds)
np.savetxt('y_pred.txt',y_pred,delimiter=',',fmt='%.0f')

acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))

# print the solutions
print("The calculated MSE error is:", error_mean_squared)
#print("The calculated mean absolute Percantage error is:", error_mean_percent)







