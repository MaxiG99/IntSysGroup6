# Add Data
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from numpy import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score
from numpy import clip, column_stack, argmax
from sklearn.datasets import load_wine


data = load_wine()
X = data.data
y = data.target
print(X)
print(y)
"""
# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets
"""
var_names = data.feature_names
var_names = [var_names[i][0:-5] for i in range(0, len(var_names))]
var_names = [var_names[i].title().replace(' ','') for i in range(0, len(var_names))]

#Split
y_1_vs_all = copy(y)
y_1_vs_all[y_1_vs_all==1] = -1
y_1_vs_all[y_1_vs_all!=-1] = 0
y_1_vs_all[y_1_vs_all==-1] = 1

y_2_vs_all = copy(y)
y_2_vs_all[y_2_vs_all==2] = -1
y_2_vs_all[y_2_vs_all!=-1] = 0
y_2_vs_all[y_2_vs_all==-1] = 1

y_3_vs_all = copy(y)
y_3_vs_all[y_3_vs_all==3] = -1
y_3_vs_all[y_3_vs_all!=-1] = 0
y_3_vs_all[y_3_vs_all==-1] = 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_1_vs_all, _ = train_test_split(X, y_1_vs_all, test_size=0.2, random_state=42)
_, _, y_train_2_vs_all, _ = train_test_split(X, y_2_vs_all, test_size=0.2, random_state=42)
_, _, y_train_3_vs_all, _ = train_test_split(X, y_3_vs_all, test_size=0.2, random_state=42)

# %% Train 1 vs all model
# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_1_vs_all, nr_clus=10)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_1_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_1_vs_all = modbuilder.get_model()