close all; clc; clearvars;

%% Load data 
cd Data
kinematics_data= load('robot_inverse_kinematics_dataset_own.csv')
cd ..

%define Variable names, angle threshold, clusters and epochs
Variables = {"q1",'q2','q3','q4','q5','q6','x','y','z'}
threshold = pi/2
clusters = 20
numfuzrules = 5
epochs = 100


% %delete negative Data-Points in the data for each angle seperately 
% columbsToCheck = [1,2,3,4,5,6]
% 
% negativeRows_q1 = any(kinematics_data(:,1) < 0 , 2)
% negativeRows_q2 = any(kinematics_data(:,2) < 0 , 2)
% negativeRows_q3 = any(kinematics_data(:,3) < 0 , 2)
% negativeRows_q4 = any(kinematics_data(:,4) < 0 , 2)
% negativeRows_q5 = any(kinematics_data(:,5) < 0 , 2)
% negativeRows_q6 = any(kinematics_data(:,6) < 0 , 2)

% dataForQ1 = kinematics_data
% dataForQ2 = kinematics_data
% dataForQ3 = kinematics_data
% dataForQ4 = kinematics_data
% dataForQ5 = kinematics_data
% dataForQ6 = kinematics_data

% dataForQ1(negativeRows_q1,:) = []
% dataForQ2(negativeRows_q2,:) = []
% dataForQ3(negativeRows_q3,:) = []
% dataForQ4(negativeRows_q4,:) = []
% dataForQ5(negativeRows_q5,:) = []
% dataForQ6(negativeRows_q6,:) = []



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
data_q1=[X(:) Y(:) Z(:) ]
data_q2=[X(:) Y(:) Z(:) q1(:)]
data_q3=[X(:) Y(:) Z(:) q1(:) q2(:)]
data_q4=[X(:) Y(:) Z(:) q1(:) q2(:) q3(:)]
data_q5=[X(:) Y(:) Z(:) q1(:) q2(:) q3(:) q4(:)]
data_q6=[X(:) Y(:) Z(:) q1(:) q2(:) q3(:) q4(:) q5(:)]

plot3(X,Y,Z,'r.')
axis equal;
xlabel('X','fontsize',10)
ylabel('Y','fontsize',10)
zlabel('Z','fontsize',10)
title('X-Y-Z coordinates for all q1-q6 combinations','fontsize',10)

%define training and test data sets for each angle that the fuzzy model
%needs to be traiend with later
train_test_partition = cvpartition(length(data_q1),'Holdout',0.3);
validation_test_partition = cvpartition(train_test_partition.TestSize,'Holdout',1/3);
train_idx = training(train_test_partition);
test_idx = training(validation_test_partition);
validation_idx = test(validation_test_partition)



training_data = kinematics_data(train_idx,:)
test_data = kinematics_data(test_idx,:)
validation_data =  kinematics_data(validation_idx,:)


% define Anfis options. Anfis command will Tune Sugeno-type fuzzy inference system using training data
% opt_A = anfisOptions;
% opt_A.InitialFIS = numfuzrules;
% opt_A.EpochNumber = epochs;
% opt_A.ValidationData = validation_data(:,[7:9 1])
% opt_A.DisplayANFISInformation = 0;
% opt_A.DisplayErrorValues = 0;
% opt_A.DisplayStepSize = 0;
% opt_A.DisplayFinalResults = 0;


% define genfis options.

opt_G = genfisOptions("SubtractiveClustering"); %,'FISType','sugeno'
% opt_G.NumClusters = clusters;
opt_G.ClusterInfluenceRange = 0.5; % for "SubtractiveClustering"

%define tunefis options
opt_T = tunefisOptions
opt_T.Method = "anfis"
opt_T.MethodOptions.InitialFIS = numfuzrules
opt_T.MethodOptions.EpochNumber = epochs
opt_T.MethodOptions.ValidationData = validation_data(:,[7:9 1])

% % training the ANFIS networks using the anfis command
genfis1 = genfis(training_data(:,7:9), training_data(:,1),opt_G)
% [in,out,rule] = getTunableSettings(genfis1)
% anfis1 = tunefis(genfis1,[in;out],training_data(:,7:9),training_data(:,1),opt_T)

genfis2 = genfis(training_data(:,[7:9 1]), training_data(:,2),opt_G)
opt_T.MethodOptions.ValidationData = validation_data(:,[7:9 1 2])
% [in,out,rule] = getTunableSettings(genfis2)
% anfis2 = tunefis(genfis2,[in;out],training_data(:,[7:9 1]),training_data(:,2),opt_T)

genfis3 = genfis(training_data(:,[7:9 1 2]), training_data(:,3),opt_G)
opt_T.MethodOptions.ValidationData = validation_data(:,[7:9 1 2 3])
% [in,out,rule] = getTunableSettings(genfis3)
% anfis3 = tunefis(genfis3,[in;out],training_data(:,[7:9 1 2]),training_data(:,3),opt_T)

genfis4 = genfis(training_data(:,[7:9 1 2 3]), training_data(:,4),opt_G)
opt_T.MethodOptions.ValidationData = validation_data(:,[7:9 1 2 3 4])
% [in,out,rule] = getTunableSettings(genfis4)
% anfis4 = tunefis(genfis4,[in;out],training_data(:,[7:9 1 2 3]),training_data(:,4),opt_T)

genfis5 = genfis(training_data(:,[7:9 1 2 3 4]), training_data(:,5),opt_G)
opt_T.MethodOptions.ValidationData = validation_data(:,[7:9 1 2 3 4 5])
% [in,out,rule] = getTunableSettings(genfis5)
% anfis5 = tunefis(genfis5,[in;out],training_data(:,[7:9 1 2 3 4]),training_data(:,5),opt_T)











% create test data with the predicted robot angles of the joints before the corresponding join and predict the output of the test data
q1_test = [test_data(:,1:3)]
q1_pred = evalfis(genfis1,q1_test)

q2_test = [test_data(:,1:3) q1_pred(:)]
q2_pred = evalfis(genfis2,q2_test)

q3_test = [test_data(:,1:3) q1_pred(:) q2_pred(:)]
q3_pred = evalfis(genfis3,q3_test)

q4_test = [test_data(:,1:3) q1_pred(:) q2_pred(:) q3_pred(:)]
q4_pred = evalfis(genfis4,q4_test)

q5_test = [test_data(:,1:3) q1_pred(:) q2_pred(:) q3_pred(:) q4_pred(:)]
q5_pred = evalfis(genfis5,q5_test)

% q6_test = [test_data(:,1:3) q1_pred(:) q2_pred(:) q3_pred(:) q4_pred(:) q5_pred(:)]
% q6_pred = evalfis(anfis6,q6_test)

% % calculate the difference between predicted and real output
q_1_diff = test_data(:,1) - q1_pred;
q_2_diff = test_data(:,2) - q2_pred;
q_3_diff = test_data(:,3) - q3_pred;
q_4_diff = test_data(:,4) - q4_pred;
q_5_diff = test_data(:,5) - q5_pred;
% q_6_diff = test_data(:,6) - q6_pred;

% % plot the difference between predicted and real output
figure
plot(q_1_diff)
ylabel('q1_Test - q1_predict')
title('Deduced q1 - Predicted q1')

figure
plot(q_2_diff)
ylabel('q2_Test - q2_predict')
% title('Deduced q2 - Predicted q2')

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

% figure
% plot(q_6_diff)
% ylabel('q6_Test - q6_predict')
% title('Deduced q6 - Predicted q6')


% % calculate mean absolute error
mae_q1 = mae(q_1_diff)
mae_q2 = mae(q_2_diff)
mae_q3 = mae(q_3_diff)
mae_q4 = mae(q_4_diff)
mae_q5 = mae(q_5_diff)
% mae_q6 = mae(q_6_diff)

% % calculate root mean square error
rms_q1 = rms(q_1_diff)
rms_q2 = rms(q_2_diff)
rms_q3 = rms(q_3_diff)
rms_q4 = rms(q_4_diff)
rms_q5 = rms(q_5_diff)
% rms_q6 = rms(q_6_diff)

cd Data

writematrix(q1_test,"eingabe_q1.csv")
writematrix(q2_test,"eingabe_q2.csv")
writematrix(q3_test,"eingabe_q3.csv")
writematrix(q4_test,"eingabe_q4.csv")
writematrix(q5_test,"eingabe_q5.csv")

writematrix(q1_pred,"q1_pred.csv")
writematrix(q2_pred,"q2_pred.csv")
writematrix(q3_pred,"q3_pred.csv")
writematrix(q4_pred,"q4_pred.csv")
writematrix(q5_pred,"q5_pred.csv")

cd ..

















