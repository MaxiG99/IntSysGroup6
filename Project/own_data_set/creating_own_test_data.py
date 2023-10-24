#%%
import numpy as np
import pandas as pd

def calculate_xyz_coordinates(joint_angles):
    # Extract joint angles
    theta1, theta2, theta3, theta4, theta5, theta6 = joint_angles

    # Define constants
    l1 = 1.0  # Replace with the actual value for l1
    l2 = 20.0  # Replace with the actual value for l2
    l3 = 30.0  # Replace with the actual value for l3
    l4 = 40.0  # Replace with the actual value for l4
    l5 = 50.0  # Replace with the actual value for l5
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
    C6 = np.cos(theta6) # not used
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


# %%

# Number of data points
num_data_points = 5000


# Define the joint value ranges
theta_min = np.deg2rad(0)
theta_max = np.deg2rad(90)


# Generate random joint values within the specified ranges
theta1_values = np.random.uniform(theta_min, theta_max, num_data_points)
theta2_values = np.random.uniform(theta_min, theta_max, num_data_points)
theta3_values = np.random.uniform(theta_min, theta_max, num_data_points)
theta4_values = np.random.uniform(theta_min, theta_max, num_data_points)
theta5_values = np.random.uniform(theta_min, theta_max, num_data_points)
theta6_values = np.random.uniform(theta_min, theta_max, num_data_points)


# Calculate XYZ coordinates for each set of joint values (you'll need to modify this for your specific robot)
joint_angles = [theta1_values,theta2_values,theta3_values,theta4_values,theta5_values,theta6_values]

x_values = []
y_values = []
z_values = []

for i in range(num_data_points):
    x, y, z = calculate_xyz_coordinates([theta1_values[i],theta2_values[i],theta3_values[i],theta4_values[i],theta5_values[i],theta6_values[i]])# change input

    #x, y, z = calculate_xyz_coordinates(joint_angles)
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

# Create a DataFrame to store the data
data = pd.DataFrame({'q1': theta1_values,
                     'q2': theta2_values,
                     'q3': theta3_values,
                     'q4': theta4_values,
                     'q5': theta5_values,
                     'q6': theta6_values,
                     'x': x_values,
                     'y': y_values,
                     'z': z_values})


# Save the data to a CSV file
path = '/Users/HP Spectre/OneDrive - student.kit.edu/uni/Master/Lissabon Kurse/Intelligent Systems/IntSysGroup6/'

data.to_csv(path+'Project/data/robot_inverse_kinematics_dataset_own.csv', index=False)
