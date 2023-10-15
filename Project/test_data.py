#%%
import numpy as np
import pandas as pd

def calculate_xyz_coordinates(joint_angles):
    # Extract joint angles
    theta1, theta2, theta3, theta4, theta5, theta6 = joint_angles

    # Define constants
    l1 = 1.0  # Replace with the actual value for l1
    l2 = 1.0  # Replace with the actual value for l2
    l3 = 1.0  # Replace with the actual value for l3
    l4 = 1.0  # Replace with the actual value for l4
    l5 = 1.0  # Replace with the actual value for l5
    l23 = l2 + l3  # Sum of l2 and l3

    # Calculate trigonometric values
    S1 = np.sin(theta1)
    C1 = np.cos(theta1)
    S2 = np.sin(theta2)
    C2 = np.cos(theta2)
    S3 = np.sin(theta3)
    C3 = np.cos(theta3)
    S4 = np.sin(theta4)
    C4 = np.cos(theta4)
    S5 = np.sin(theta5)
    C5 = np.cos(theta5)
    S6 = np.sin(theta6)
    C6 = np.cos(theta6)
    C23 = np.cos(theta2+theta3)
    S23 = np.sin(theta2+theta3)

    # Calculate the individual transformation matrix elements
    #T00 = ((S1 * S4 + C1 * C4 * C23) * C5 - S5 * S23 * C1) * C6 + (S1 * C4 - S4 * C1 * C23) * S6
    #T10 = ((S1 * C4 * C23 - S4 * C1) * C5 - S1 * S5 * S23) * C6 - (S1 * S4 * C23 + C1 * C4) * S6
    #T20 = -(S5 * C23 + S23 * C4 * C5) * C6 + S4 * S6 * S23
    #T30 = 0
    #T01 = (-(S1 * S4 + C1 * C4 * C23) * C5 + S5 * S23 * C1) * S6 + (S1 * C4 - S4 * C1 * C23) * C6
    #T11 = (-S1 * C4 * C23 + S4 * C1) * C5 + S1 * S5 * S23 * S6 - (S1 * S4 * C23 + C1 * C4) * C6
    #T21 = (S5 * C23 + S23 * C4 * C5) * S6 + S4 * S23 * C6
    #T31 = 0
    #T02 = -(S1 * S4 + C1 * C4 * C23) * S5 - S23 * C1 * C5
    #T12 = (-S1 * C4 * C23 + S4 * C1) * S5 - S1 * S23 * C5
    #T22 = S5 * S23 * C4 - C5 * C23
    #T32 = 0
    T03 = -l4 * (S1 * S4 * S5 + S5 * C1 * C4 * C23 + S23 * C1 * C5) - l1 * S23 * C1 + l3 * C1 * C2 + l5 * C1 * C23
    T13 = -l4 * (S1 * S5 * C4 * C23 + S1 * S23 * C5) - l1 * S1 * S23 + l3 * S1 * C2 + l5 * S1 * C23 + l4 * S4 * S5 * C1
    T23 = -l3 * S2 + l4 * (S5 * S23 * C4 - C5 * C23) + l5 * S23 + l4 * C5 * C23 - l1 * C23 + l2
    T33 = 1

    # Extract the XYZ coordinates
    x = T03
    y = T13
    z = T23

    return x, y, z

# Example usage:
joint_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
x, y, z = calculate_xyz_coordinates(joint_angles)
print("X:", x)
print("Y:", y)
print("Z:", z)


#%%
path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data = pd.read_csv(path+'Project/data/robot_inverse_kinematics_dataset.csv')
print(data.head()) # joint angels in rad

# Splitting into Output Joint angels q and input cartesian coordinates xyz
q = data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']]
xyz = data[['x','y','z']]

df = pd.DataFrame(q)
q_first_row = df.iloc[0]
x, y, z = calculate_xyz_coordinates(q_first_row)
print("X:", x)
print("Y:", y)
print("Z:", z)
df = pd.DataFrame(xyz)
xyz_first_row = df.iloc[0]
print(xyz_first_row)
# %%
