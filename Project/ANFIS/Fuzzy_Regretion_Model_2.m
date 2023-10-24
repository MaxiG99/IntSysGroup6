close all; clc; clearvars;

%% Load data 
cd Data
kinematics_data= load('robot_inverse_kinematics_dataset_own.csv')
cd ..

%define Variable names, angle threshold, clusters and epochs
Variables = {"q1",'q2','q3','q4','q5','q6','x','y','z'}
threshold = pi/2
clusters = 3
epochs = 70


%delete negative Data-Points in the data for each angle seperately 
columbsToCheck = [1,2,3,4,5,6]

negativeRows_q1 = any(kinematics_data(:,1) < 0 , 2)
negativeRows_q2 = any(kinematics_data(:,2) < 0 , 2)
negativeRows_q3 = any(kinematics_data(:,3) < 0 , 2)
negativeRows_q4 = any(kinematics_data(:,4) < 0 , 2)
negativeRows_q5 = any(kinematics_data(:,5) < 0 , 2)
negativeRows_q6 = any(kinematics_data(:,6) < 0 , 2)

dataForQ1 = kinematics_data
dataForQ2 = kinematics_data
dataForQ3 = kinematics_data
dataForQ4 = kinematics_data
dataForQ5 = kinematics_data
dataForQ6 = kinematics_data

dataForQ1(negativeRows_q1,:) = []
dataForQ2(negativeRows_q2,:) = []
dataForQ3(negativeRows_q3,:) = []
dataForQ4(negativeRows_q4,:) = []
dataForQ5(negativeRows_q5,:) = []
dataForQ6(negativeRows_q6,:) = []

%delete angle values above 90° (pi/2) since 

%define Variable data out of the given full data set
X=kinematics_data(:,7)
Y=kinematics_data(:,8)
Z=kinematics_data(:,9)
q1=kinematics_data(:,1)
q2=kinematics_data(:,2)
q3=kinematics_data(:,3)
q4=kinematics_data(:,4)
q5=kinematics_data(:,5)
q6=kinematics_data(:,6)


%define data sets for each output variable (output: q1,q2,q3,q4,q5,q6) with the input variables (input: X,Y,Z)
%needed for training later on
data_q1=[dataForQ1(:,7) dataForQ1(:,8) dataForQ1(:,9) dataForQ1(:,1)]
data_q2=[dataForQ2(:,7) dataForQ2(:,8) dataForQ2(:,9) dataForQ2(:,2)]
data_q3=[dataForQ3(:,7) dataForQ3(:,8) dataForQ3(:,9) dataForQ3(:,3)]
data_q4=[dataForQ4(:,7) dataForQ4(:,8) dataForQ4(:,9) dataForQ4(:,4)]
data_q5=[dataForQ5(:,7) dataForQ5(:,8) dataForQ5(:,9) dataForQ5(:,5)]
data_q6=[dataForQ6(:,7) dataForQ6(:,8) dataForQ6(:,9) dataForQ6(:,6)]

plot3(X,Y,Z,'r.')
axis equal;
xlabel('X','fontsize',10)
ylabel('Y','fontsize',10)
zlabel('Z','fontsize',10)
title('X-Y-Z coordinates for all q1-q6 combinations','fontsize',10)

%define training and test data sets for each angle that the fuzzy model
%needs to be traiend with later
train_test_partition_q1 = cvpartition(length(data_q1),'Holdout',0.3);
validation_test_partition_q1 = cvpartition(train_test_partition_q1.TestSize,'Holdout',1/3);
train_idx_q1 = training(train_test_partition_q1);
test_idx_q1 = training(validation_test_partition_q1);
validation_idx_q1 = test(validation_test_partition_q1)

train_test_partition_q2 = cvpartition(length(data_q2),'Holdout',0.3);
validation_test_partition_q2 = cvpartition(train_test_partition_q2.TestSize,'Holdout',1/3);
train_idx_q2 = training(train_test_partition_q2);
test_idx_q2 = training(validation_test_partition_q2);
validation_idx_q2 = test(validation_test_partition_q2)

train_test_partition_q3 = cvpartition(length(data_q3),'Holdout',0.3);
validation_test_partition_q3 = cvpartition(train_test_partition_q3.TestSize,'Holdout',1/3);
train_idx_q3 = training(train_test_partition_q3);
test_idx_q3 = training(validation_test_partition_q3);
validation_idx_q3 = test(validation_test_partition_q3)

train_test_partition_q4 = cvpartition(length(data_q4),'Holdout',0.3);
validation_test_partition_q4 = cvpartition(train_test_partition_q4.TestSize,'Holdout',1/3);
train_idx_q4 = training(train_test_partition_q4);
test_idx_q4 = training(validation_test_partition_q4);
validation_idx_q4 = test(validation_test_partition_q4)

train_test_partition_q5 = cvpartition(length(data_q5),'Holdout',0.3);
validation_test_partition_q5 = cvpartition(train_test_partition_q5.TestSize,'Holdout',1/3);
train_idx_q5 = training(train_test_partition_q5);
test_idx_q5 = training(validation_test_partition_q5);
validation_idx_q5 = test(validation_test_partition_q5)

train_test_partition_q6 = cvpartition(length(data_q6),'Holdout',0.3);
validation_test_partition_q6 = cvpartition(train_test_partition_q6.TestSize,'Holdout',1/3);
train_idx_q6 = training(train_test_partition_q6);
test_idx_q6 = training(validation_test_partition_q6);
validation_idx_q6 = test(validation_test_partition_q6)


q_1_training = data_q1(train_idx_q1,:)
q_1_test = data_q1(test_idx_q1,:)
q_1_validation =  data_q1(validation_idx_q1,:)

q_2_training = data_q2(train_idx_q2,:)
q_2_test = data_q2(test_idx_q2,:)
q_2_validation =  data_q2(validation_idx_q2,:)

q_3_training = data_q3(train_idx_q3,:)
q_3_test = data_q3(test_idx_q3,:)
q_3_validation =  data_q3(validation_idx_q3,:)

q_4_training = data_q4(train_idx_q4,:)
q_4_test = data_q4(test_idx_q4,:)
q_4_validation =  data_q4(validation_idx_q4,:)

q_5_training = data_q5(train_idx_q5,:)
q_5_test = data_q5(test_idx_q5,:)
q_5_validation =  data_q5(validation_idx_q5,:)

q_6_training = data_q6(train_idx_q6,:)
q_6_test = data_q6(test_idx_q6,:)
q_6_validation =  data_q6(validation_idx_q6,:)

 
% define Anfis options. Anfis command will Tune Sugeno-type fuzzy inference system using training data
opt_A = anfisOptions;
opt_A.InitialFIS = clusters;
opt_A.EpochNumber = epochs;
opt_A.ValidationData = q_1_validation
% opt_A.DisplayANFISInformation = 0;
% opt_A.DisplayErrorValues = 0;
% opt_A.DisplayStepSize = 0;
% opt_A.DisplayFinalResults = 0;
% 
% % define genfis options. Genfis will Generate fuzzy inference system object from data
% trying both to see which is better
% 
% opt_G = genfisOptions('FCMClustering','FISType','sugeno');
% opt_G.NumClusters = clusters;

% training the ANFIS networks using the anfis command
anfis1 = anfis(q_1_training,opt_A)
opt_A.ValidationData = q_2_validation
anfis2 = anfis(q_2_training,opt_A)
opt_A.ValidationData = q_3_validation
anfis3 = anfis(q_3_training,opt_A)
opt_A.ValidationData = q_4_validation
anfis4 = anfis(q_4_training,opt_A)
opt_A.ValidationData = q_5_validation
anfis5 = anfis(q_5_training,opt_A)
opt_A.ValidationData = q_6_validation
anfis6 = anfis(q_6_training,opt_A)

% predict the output of the test data
q1_pred = evalfis(anfis1,q_1_test(:,1:3))
q2_pred = evalfis(anfis2,q_2_test(:,1:3))
q3_pred = evalfis(anfis3,q_3_test(:,1:3))
q4_pred = evalfis(anfis4,q_4_test(:,1:3))
q5_pred = evalfis(anfis5,q_5_test(:,1:3))
q6_pred = evalfis(anfis6,q_6_test(:,1:3))

% calculate the difference between predicted and real output
q_1_diff = q_1_test(:,4) - q1_pred;
q_2_diff = q_2_test(:,4) - q2_pred;
q_3_diff = q_3_test(:,4) - q3_pred;
q_4_diff = q_4_test(:,4) - q4_pred;
q_5_diff = q_5_test(:,4) - q5_pred;
q_6_diff = q_6_test(:,4) - q6_pred;

% plot the difference between predicted and real output
figure
plot(q_1_diff)
ylabel('q1_Test - q1_predict')
title('Deduced q1 - Predicted q1')

figure
plot(q_2_diff)
ylabel('q2_Test - q2_predict')
title('Deduced q2 - Predicted q2')

figure
plot(q_3_diff)
ylabel('q3_Test - q3_predict')
title('Deduced q3 - Predicted q3')

figure
plot(q_4_diff)
ylabel('q4_Test - q4_predict')
title('Deduced q4 - Predicted q4')

figure
plot(q_5_diff)
ylabel('q5_Test - q5_predict')
title('Deduced q5 - Predicted q5')

figure
plot(q_6_diff)
ylabel('q6_Test - q6_predict')
title('Deduced q6 - Predicted q6')


% calculate mean absolute error
mae_q1 = mae(q_1_diff)
mae_q2 = mae(q_2_diff)
mae_q3 = mae(q_3_diff)
mae_q4 = mae(q_4_diff)
mae_q5 = mae(q_5_diff)
mae_q6 = mae(q_6_diff)

% calculate root mean square error
rms_q1 = rms(q_1_diff)
rms_q2 = rms(q_2_diff)
rms_q3 = rms(q_3_diff)
rms_q4 = rms(q_4_diff)
rms_q5 = rms(q_5_diff)
rms_q6 = rms(q_6_diff)





















