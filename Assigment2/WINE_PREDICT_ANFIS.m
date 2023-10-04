%Load ANFIS
anfis_model_0_vs_all = readfis('anfis_model_0_vs_all.fis');
anfis_model_1_vs_all = readfis('anfis_model_1_vs_all.fis');
anfis_model_2_vs_all = readfis('anfis_model_2_vs_all.fis');

%Put in new Wine Data that we want to classify 
new_data = [];

Y_pred_final_0_vs_all = evalfis(anfis_model_0_vs_all, new_data);
Y_pred_final_0_vs_all(Y_pred_final_0_vs_all<0) = 0;
Y_pred_final_0_vs_all(Y_pred_final_0_vs_all>1) = 1;

Y_pred_final_1_vs_all = evalfis(anfis_model_1_vs_all, new_data);
Y_pred_final_1_vs_all(Y_pred_final_1_vs_all<0) = 0;
Y_pred_final_1_vs_all(Y_pred_final_1_vs_all>1) = 1;

Y_pred_final_2_vs_all = evalfis(anfis_model_2_vs_all, new_data);
Y_pred_final_2_vs_all(Y_pred_final_2_vs_all<0) = 0;
Y_pred_final_2_vs_all(Y_pred_final_2_vs_all>1) = 1;

Y_pred_final = zeros(size(new_data,1),3);
Y_pred_final(:,1) = Y_pred_final_0_vs_all + (1-Y_pred_final_1_vs_all) + (1-Y_pred_final_2_vs_all);
Y_pred_final(:,2) = Y_pred_final_1_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_2_vs_all);
Y_pred_final(:,3) = Y_pred_final_2_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_1_vs_all);
Y_pred_final = Y_pred_final./sum(Y_pred_final,2);
[~,Y_pred_final_labels] = max(Y_pred_final,[],2);
Y_pred_final_labels = Y_pred_final_labels - 1;
