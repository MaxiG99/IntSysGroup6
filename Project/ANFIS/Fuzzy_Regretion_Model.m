close all; clc; clearvars;

%% Load data 
cd Data
kinematics_data= load('robot_inverse_kinematics_dataset.csv')
cd ..

%define Variable names
Variables = {"q1",'q2','q3','q4','q5','q6','x','y','z'}
clusters = 3
epochs = 200


%delete negative Data-Points for each angle seperately
clumbsToCheck = [1,2,3,4,5,6]

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

%define Variable data out of the given data set
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
data_q1=[X(:) Y(:) Z(:) q1(:)]
data_q2=[X(:) Y(:) Z(:) q2(:)]
data_q3=[X(:) Y(:) Z(:) q3(:)]
data_q4=[X(:) Y(:) Z(:) q4(:)]
data_q5=[X(:) Y(:) Z(:) q5(:)]
data_q6=[X(:) Y(:) Z(:) q6(:)]


plot3(X,Y,Z,'r.')
axis equal;
xlabel('X','fontsize',10)
ylabel('Y','fontsize',10)
zlabel('Z','fontsize',10)
title('X-Y-Z coordinates for all q1-q6 combinations','fontsize',10)

%define training and test data sets
train_test_partition = cvpartition(15000,'Holdout',0.3);
validation_test_partition = cvpartition(4500,'Holdout',1500);

train_idx = training(train_test_partition);
test_idx = training(validation_test_partition);
validation_idx = test(validation_test_partition)

q_1_training = data_q1(train_idx,:)
q_1_test = data_q1(test_idx,:)
q_1_validation =  data_q1(validation_idx,:)

q_2_training = data_q2(train_idx,:)
q_2_test = data_q2(test_idx,:)

q_3_training = data_q3(train_idx,:)
q_3_test = data_q3(test_idx,:)

q_4_training = data_q4(train_idx,:)
q_4_test = data_q4(test_idx,:)

q_5_training = data_q5(train_idx,:)
q_5_test = data_q5(test_idx,:)

q_6_training = data_q6(train_idx,:)
q_6_test = data_q6(test_idx,:)


%define Anfis options. Anfis command will Tune Sugeno-type fuzzy inference system using training data
opt_A = anfisOptions;
opt_A.InitialFIS = clusters;
opt_A.EpochNumber = epochs;
opt_A.ValidationData = q_1_validation
% opt_A.DisplayANFISInformation = 0;
% opt_A.DisplayErrorValues = 0;
% opt_A.DisplayStepSize = 0;
% opt_A.DisplayFinalResults = 0;

%define genfis options. Genfis will Generate fuzzy inference system object from data
%trying both to see which is better

opt_G = genfisOptions('FCMClustering','FISType','sugeno');
opt_G.NumClusters = clusters;

%training the ANFIS networks using the anfis command
anfis1 = anfis(q_1_training,opt_A)
% anfis2 = anfis(q_2_training,opt_A)
% anfis3 = anfis(q_3_training,opt_A)
% anfis4 = anfis(q_4_training,opt_A)
% anfis5 = anfis(q_5_training,opt_A)
% anfis6 = anfis(q_6_training,opt_A)

%predict the output of the test data
q1_pred = evalfis(anfis1,q_1_test(:,1:3))
% q2_pred = evalfis(anfis1,q_2_test(:,1:3))
% q3_pred = evalfis(anfis1,q_3_test(:,1:3))
% q4_pred = evalfis(anfis1,q_4_test(:,1:3))
% q5_pred = evalfis(anfis1,q_5_test(:,1:3))
% q6_pred = evalfis(anfis1,q_6_test(:,1:3))

%calculate the difference between predicted and real output
q_1_diff = q_1_test(:,4) - q1_pred;
% q_2_diff = q_2_test(:,4) - q2_pred;
% q_3_diff = q_3_test(:,4) - q3_pred;
% q_4_diff = q_4_test(:,4) - q4_pred;
% q_5_diff = q_5_test(:,4) - q5_pred;
% q_6_diff = q_6_test(:,4) - q6_pred;

%plot the difference between predicted and real output
figure
plot(q_1_diff)
ylabel('q1_Test - q1_predict')
title('Deduced q1 - Predicted q1')
% 
% figure
% plot(q_2_diff)
% ylabel('q2_Test - q2_predict')
% title('Deduced q2 - Predicted q2')
% 
% figure
% plot(q_3_diff)
% ylabel('q3_Test - q3_predict')
% title('Deduced q3 - Predicted q3')
% 
% figure
% plot(q_4_diff)
% ylabel('q4_Test - q4_predict')
% title('Deduced q4 - Predicted q4')
% 
% figure
% plot(q_5_diff)
% ylabel('q5_Test - q5_predict')
% title('Deduced q5 - Predicted q5')
% 
% figure
% plot(q_6_diff)
% ylabel('q6_Test - q6_predict')
% title('Deduced q6 - Predicted q6')


%calculate mean absolute error
mae_q1 = mae(q_1_diff)
% mae_q2 = mae(q_2_diff)
% mae_q3 = mae(q_3_diff)
% mae_q4 = mae(q_4_diff)
% mae_q5 = mae(q_5_diff)
% mae_q6 = mae(q_6_diff)

%calculate root mean square error
rms_q1 = rms(q_1_diff)
% rms_q2 = rms(q_2_diff)
% rms_q3 = rms(q_3_diff)
% rms_q4 = rms(q_4_diff)
% rms_q5 = rms(q_5_diff)
% rms_q6 = rms(q_6_diff)





















