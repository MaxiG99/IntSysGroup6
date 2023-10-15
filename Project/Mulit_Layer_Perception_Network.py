import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data = pd.read_csv(path+'Project/data/robot_inverse_kinematics_dataset.csv')
print(data.head()) # joint angels in rad

# Splitting into Output Joint angels q and input cartesian coordinates xyz
q = data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']]
xyz = data[['x','y','z']]
# use MinMaxScaler
scaler = MinMaxScaler()
q_s = scaler.fit_transform(q)
xyz_s = scaler.fit_transform(xyz)

#scaler inverse transform
q_train_val,q_test,  xyz_train_val, xyz_test = train_test_split(q_s ,xyz_s ,shuffle=True,train_size=0.8)

q_train, q_val, xyz_train, xyz_val = train_test_split(q_train_val,xyz_train_val,train_size=0.9)

for joint, cor in [(q_test,xyz_test),(q_train,xyz_train),(q_val,xyz_val)]:
    joint_name = [name for name, value in locals().items() if value is joint][0]
    cor_name = [name for name, value in locals().items() if value is cor][0]
    print(cor_name + " shape: " + str(cor.shape))
    print(joint_name + " shape: " + str(joint.shape))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100,100),
            nn.Tanh(),
            nn.Linear(100,6)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out
    
model = MLP()
print(model)

# Hyperparameters
learning_rate = 0.1
batch_size = 4*128
epochs = 50
weight_decay_L2 =1e-5

# Specifying Loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay_L2)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min')


# Creating Tensors of the Data 
q_train_tensor = torch.tensor(q_train, dtype=torch.float32)
q_val_tensor = torch.tensor(q_val, dtype=torch.float32)
q_test_tensor = torch.tensor(q_test, dtype=torch.float32)
xyz_train_tensor = torch.tensor(xyz_train, dtype=torch.float32)
xyz_val_tensor = torch.tensor(xyz_val, dtype=torch.float32)
xyz_test_tensor = torch.tensor(xyz_test, dtype=torch.float32)

# Start Training and Validating

train_dataset = TensorDataset(xyz_train_tensor,q_train_tensor)
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size)
val_dataset = TensorDataset(xyz_val_tensor, q_val_tensor)
val_loader = DataLoader(val_dataset, shuffle=False)

# Defining Array for plotting val and train loss after loop
loss_afer_epoch_train = []
loss_afer_epoch_val = []


class EarlyStopper:
    '''
    If Validation loss is increasing stop the return false. 
    Add patience of how many Epochs to wait.
    '''
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f"Validation loss increased. Minimum validation loss: {self.min_validation_loss:.4f}, Current validation loss: {validation_loss:.4f}")
                return True
        return False

# Creating Instance to early stop in the loop 
early_stopper = EarlyStopper(patience=0,min_delta=0.01)

for epoch in range(epochs):
    train_losses = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.float()        

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    print('Epoch %d | Training Loss %6.2f' % (epoch, sum(train_losses)/len(train_losses)))
    loss_afer_epoch_train.append(sum(train_losses)/len(train_losses))

    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch_num, val_data in enumerate(val_loader):
            val_x, val_y = val_data
            val_x = val_x.float()

            val_output = model(val_x)
            val_loss = criterion(val_output, val_y)
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    loss_afer_epoch_val.append(avg_val_loss)

    print('Epoch %d | Validation Loss %6.2f' % (epoch, avg_val_loss))
    # check if we should stop early 
    if early_stopper.early_stop(avg_val_loss):         
        print('Early break after '+str(epoch)+'Epochs')
        #break

    # adjust learning rate
    scheduler.step(avg_val_loss)


test_dataset = TensorDataset(xyz_test_tensor, q_test_tensor)
test_loader = DataLoader(test_dataset, shuffle=False)

# Evaluate the model on the test set
model.eval()
test_losses = []

joint_errors = [[] for _ in range(6)]

with torch.no_grad():
    for test_data in test_loader:
        test_xyz, test_q = test_data
        test_xyz = test_xyz.float()

        test_output = model(test_xyz)
        test_loss = criterion(test_output, test_q)
        test_losses.append(test_loss.item())

        for i in range(6):
            vector_loss = (test_output[0][i]-test_q[0][i])
            joint_errors[i].append(abs(vector_loss))

MAE= np.mean(joint_errors,axis=1)
for i in range(6):
    print("Joint "+str(i+1)+" has a MAE of "+str(MAE[i]))

# Calculate the Mean Squared Error (MSE) for the test set
test_mse = sum(test_losses) / len(test_losses)
#print(np.mean(vector_loss,axis = 1))
print('Test MSE: %6.2f' % test_mse)


# Plot the training and validation loss
plt.figure()
plt.plot(loss_afer_epoch_train, label='Training Loss')
plt.plot(loss_afer_epoch_val, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
