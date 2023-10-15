import pandas as pd
import torch 
import torch.nn as nn 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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

q_train_val,q_test,  xyz_train_val, xyz_test = train_test_split(q_s,xyz_s,random_state=42,shuffle=True,train_size=0.8)

q_train, q_val, xyz_train, xyz_val = train_test_split(q_train_val,xyz_train_val,train_size=0.9)

for joint, cor in [(q_test,xyz_test),(q_train,xyz_train),(q_val,xyz_val)]:
    joint_name = [name for name, value in locals().items() if value is joint][0]
    cor_name = [name for name, value in locals().items() if value is cor][0]
    print(cor_name + " shape: " + str(cor.shape))
    print(joint_name + " shape: " + str(joint.shape))



# Define the RNN model class
class TwoLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TwoLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x,batch_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Define input, hidden, and output dimensions
input_size = 3  # Input features
hidden_size = 32  # Hidden state size
num_layers = 2  # Number of RNN layers
output_size = 6  # Output size (for regression or classification)
batch_size = 100


q_train_tensor = torch.tensor(q_train, dtype=torch.float32)
q_val_tensor = torch.tensor(q_val, dtype=torch.float32)
q_test_tensor = torch.tensor(q_test, dtype=torch.float32)
xyz_train_tensor = torch.tensor(xyz_train, dtype=torch.float32)
xyz_val_tensor = torch.tensor(xyz_val, dtype=torch.float32)
xyz_test_tensor = torch.tensor(xyz_test, dtype=torch.float32)


train_dataset = TensorDataset(xyz_train_tensor,q_train_tensor)
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size)

# Define your RNN model
model = TwoLayerRNN(input_size, hidden_size, num_layers, output_size)
# Print the model architecture
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
learning_rate = 0.1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs,batch_size)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()










