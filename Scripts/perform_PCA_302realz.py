# PCA on 302 REALZ DATA (see perform_PCA.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Create and save encoded permeability and potential data
# Perform PCA analysis with variance of 0.99
# Save encoded components as *.csv files
# PCA COMPONENTS EXTRACTION AND RECONSTRUCTION ERRORS:
#     PERMEABILITY: ln[k] --> no normalization --> extract PCA perm comps
#     ELECTRICAL POTENTIAL: NormByMeasureSMSD  --> extract PCA comps
#     PCA RECONSTRUCTION ERRORS: Permeability and potential data
# AUTHOR: Maruti Kumar Mudunuru

import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Start time
start_time = time.time()
#=============================================================================;
#  Function-1: Plot PCA reconstruction error heat maps (perm and potentials)  ;
#=============================================================================;
def plot_heatmap(data_list, str_x_label, str_y_label, str_fig_name, \
	             v_min, v_max):

    #---------------------------------------------------;
    #  Heatmap of the error data (min, max, mean, std)  ;
    #---------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    #plt.grid(True)
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([0, 730])
    #ax.set_ylim([0, 80])
    cmap = 'bwr'
    sns.heatmap(data_list, vmin = v_min, vmax = v_max, cmap = cmap)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #sns.heatmap(data_list, x_axis_labels, y_axis_labels)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.legend(loc = 'upper center')
    fig.tight_layout()
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#=============================================================================;
#  Function-2: Plot PCA reconstruction error histogram (perm and potentials)  ;
#=============================================================================;
def plot_histplot(data_list, str_x_label, str_y_label, str_fig_name):

    #---------------------------------------------------;
    #  Heatmap of the error data (min, max, mean, std)  ;
    #---------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    #plt.grid(True)
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([0, 730])
    #ax.set_ylim([0, 80])
    sns.histplot(data_list, bins = 5)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #sns.heatmap(data_list, x_axis_labels, y_axis_labels)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.legend(loc = 'upper center')
    fig.tight_layout()
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#============================================================;
#  Function-3: Plot one-to-one for train/val/test data       ;
#              (Ground truth vs. PCA reconstruction errors)  ; 
#============================================================;
def plot_gt_pred(x, y, str_x_label, str_y_label, str_fig_name):

    #------------------------------------------------;
    #  Plot one-to-one (ground truth vs. predicted)  ;
    #------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)   
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    min_val = np.min(x)
    max_val = np.max(x)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    sns.lineplot([min_val, max_val], [min_val, max_val], \
                  linestyle = 'solid', linewidth = 1.5, \
                  color = 'r') #One-to-One line
    sns.scatterplot(x, y, color = 'b', marker = 'o')
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(str_fig_name + '.png')

#*************************************************************;
# 0. Set OLD data paths (pre-processed and normalized *.npy)  ; 
#*************************************************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
path_ss_pca_sav      = path + "302_Realz_Models/1_SS_PCA_models/" #Saved StandardScalar and PCA models
path_PCA_pred_npy    = path + "302_Realz_Models/5_PCA_Predictions_Data/" #PCA inverse transform (reconstructed) data
path_DNN_pred_sav    = path + "302_Realz_Models/3_DL_Predictions_Data/"

#************************************************;
#  1. Get pre-processed data (all realizations)  ;
#     POTENTIAL DATA: Normalized by measure      ;
#     ln[K] DATA: Un-scaled and SS-based data    ;
#************************************************;
random_seed        = 1337
np.random.seed(random_seed) #Set random seed
#
pot_data           = np.load(path_proccessed_data + \
							"1_Normalized/NormByMeas_Potential_302_1133048.npy") #Norm by measure potential data (302,1133048)
perm_unscaled_data = np.load(path_proccessed_data + \
							"0_Raw/ln_Permeability_302_585453.npy") #Unscaled ln[K] data (302,585453)

#**********************************************************************************************;
#  2. Split into train/val/test data (scaled and unscaled ln[perm]; norm-by-measure for pots)  ;
#**********************************************************************************************;
train_index_list       = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
										"Train_Realz.txt", dtype = int, skip_header = 1) #Train indices  
val_index_list         = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
										"Val_Realz.txt", dtype = int, skip_header = 1) #Val indices
test_index_list        = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
										"Test_Realz.txt", dtype = int, skip_header = 1) #Test indices
# 
#  
train_pot              = pot_data[train_index_list-1,:] #Training (Norm by measure) potential data (282, 1133048)
val_pot                = pot_data[val_index_list-1,:] #Validation (Norm by measure) potential data (10, 1133048)
test_pot               = pot_data[test_index_list-1,:] #Test (Norm by measure) potential data (10, 1133048)
# 
train_perm_us          = perm_unscaled_data[train_index_list-1,:] #Training unscaled ln[K] data (282, 585453)
val_perm_us            = perm_unscaled_data[val_index_list-1,:] #Validation unscaled ln[K] data (10, 585453)
test_perm_us           = perm_unscaled_data[test_index_list-1,:] #Test unscaled ln[K] data (10, 585453)
#
np.save(path_DNN_pred_sav + "0_Raw_TrainValTest/train_perm_us.npy", train_perm_us) #Save train perm GT
np.save(path_DNN_pred_sav + "0_Raw_TrainValTest/val_perm_us.npy", val_perm_us) #Save val perm GT
np.save(path_DNN_pred_sav + "0_Raw_TrainValTest/test_perm_us.npy", test_perm_us) #Save test perm GT

#********************************************************************************;
#  3. PCA model initialization, save, and training  (explained variance = 0.99)  ;
#********************************************************************************;
potential_pca = PCA(0.99) #potential PCA                        --> give 270 pots PCA components
perm_us_pca   = PCA(0.99) #perm_us_pca for un-scaled ln[K] data --> give 246 perm PCA components
#
potential_pca.fit(train_pot) #Fit PCA for potential training data (Norm by Measure SMSD)
perm_us_pca.fit(train_perm_us) #Fit PCA for ln[K] training data (Un-scaled)
#
pot_pca_name     = path_ss_pca_sav + "1_PCA/potential_pca_model.sav" #Potential PCA fit on training data (Norm by Measure SMSD)
perm_us_pca_name = path_ss_pca_sav + "1_PCA/perm_pca_unscaled_model.sav" #ln[K] fit on training data (Un-scaled)
pickle.dump(potential_pca, open(pot_pca_name, 'wb')) #Save the fitted PCA potential model (Norm by Measure SMSD)
pickle.dump(perm_us_pca, open(perm_us_pca_name, 'wb')) #Save the fitted PCA permeability model (Un-scaled)
#
explained_potential_var = potential_pca.explained_variance_ratio_ #explained variance ratio for potential PCA data (270,)
df_ex_pot_var           = pd.DataFrame(explained_potential_var) #[270 rows x 1 columns]
df_ex_pot_var.to_csv(path_proccessed_data + "2_PCA_comps/potential_explained_ratio_variance.csv", \
                     header=None) #Save explained variance ratio of potential PCA data (270,)
#
explained_perm_us_var = perm_us_pca.explained_variance_ratio_ #explained variance ratio for unscaled ln[K] data (246,)
df_ex_perm_us_var     = pd.DataFrame(explained_perm_us_var) #[246 rows x 1 columns]
df_ex_perm_us_var.to_csv(path_proccessed_data + "2_PCA_comps/perm_unscaled_explained_ratio_variance.csv", \
                         header=None) #Save explained variance ratio of unscaled ln[K]  PCA data (246,)

#********************************************************************;
#  4a. PCA transform of train/val/test perm and pot data             ;
#       (246 perm PCA components --> unscaled perm data              ; 
#        270 pots PCA components --> norm-by-measure SMSD pot data)  ;
#********************************************************************;
potential_pca_model = pickle.load(open(pot_pca_name, 'rb')) #Load already created PCA potential model
perm_us_pca_model   = pickle.load(open(perm_us_pca_name, 'rb')) #Load already created PCA ln[K] model (Unscaled)
#
train_comp_pot      = potential_pca_model.transform(train_pot) #Get PCA components of training potential data (282, 270)
val_comp_pot        = potential_pca_model.transform(val_pot) #Get PCA components of validation potential data (10, 270)
test_comp_pot       = potential_pca_model.transform(test_pot) #Get PCA components of potential potential data (10, 270)
#
train_comp_perm_us  = perm_us_pca_model.transform(train_perm_us) #Get PCA components of training unscaled perm data (282, 246)
val_comp_perm_us    = perm_us_pca_model.transform(val_perm_us) #Get PCA components of validation unscaled perm data (10, 246)
test_comp_perm_us   = perm_us_pca_model.transform(test_perm_us) #Get PCA components of test unscaled perm data (10, 246)
#
num_pot_comps       = train_comp_pot.shape[1] #270 (No. of PCA comps of potential field)
num_perm_us_comps   = train_comp_perm_us.shape[1] #246 (No. of PCA comps of unscaled permeability field)

#***************************************************************************;
#  4b. Save PCA reconstruted permeability train/val/test data               ;
#      (unscaled PCA perm comps = 246 and PCA perm comps = 0.99 var ratio)  ;
#***************************************************************************;
train_perm_us_it    = perm_us_pca_model.inverse_transform(train_comp_perm_us) #Inverse transform (training unscaled perm) (282, 585453)
val_perm_us_it      = perm_us_pca_model.inverse_transform(val_comp_perm_us) #Inverse transform (validation unscaled perm) (10, 585453)
test_perm_us_it     = perm_us_pca_model.inverse_transform(test_comp_perm_us) #Inverse transform (testing unscaled perm) (10, 585453)
#
np.save(path_PCA_pred_npy + "train_99p_exvar.npy", train_perm_us_it) #Save train perm PCA IT exp.var.ratio = 0.99
np.save(path_PCA_pred_npy + "val_99p_exvar.npy", val_perm_us_it) #Save val perm PCA IT exp.var.ratio = 0.99
np.save(path_PCA_pred_npy + "test_99p_exvar.npy", test_perm_us_it) #Save test perm PCA IT exp.var.ratio = 0.99

#*****************************************;
#  5. Save train/val/test PCA components  ;
#*****************************************;
pot_comps_cols_list     = ["pot_pca_" + str(i) for i in range(0,num_pot_comps)] #Create pd dataframe header list (Pot PCA comps)
perm_comps_us_cols_list = ["perm_pca_" + str(i) for i in range(0,num_perm_us_comps)] #Create pd dataframe header list (Perm PCA comps) Un-scaled
#
train_comp_pot_df       = pd.DataFrame(train_comp_pot, index=train_index_list, \
                                       columns = pot_comps_cols_list) # Create pd dataframe for training pot PCA comps (282, 270)
val_comp_pot_df         = pd.DataFrame(val_comp_pot, index=val_index_list, \
                                       columns = pot_comps_cols_list) # Create pd dataframe for validation pot PCA comps (10, 270)
test_comp_pot_df        = pd.DataFrame(test_comp_pot, index=test_index_list, \
                                       columns = pot_comps_cols_list) # Create pd dataframe for test pot PCA comps (10, 270)
#
train_comp_perm_us_df   = pd.DataFrame(train_comp_perm_us, index=train_index_list, \
                                       columns = perm_comps_us_cols_list) # Create pd dataframe for training un-scaled perm PCA comps (282, 246)
val_comp_perm_us_df     = pd.DataFrame(val_comp_perm_us, index=val_index_list, \
                                       columns = perm_comps_us_cols_list) # Create pd dataframe for validation un-scaled perm PCA comps (10, 246)
test_comp_perm_us_df    = pd.DataFrame(test_comp_perm_us, index=test_index_list, \
                                       columns = perm_comps_us_cols_list) # Create pd dataframe for test un-scaled perm PCA comps (10, 246)
#
train_comp_pot_df.to_csv(path_proccessed_data + "2_PCA_comps/train_pca_comp_potential.csv") #Save train PCA comp pots (282, 270)
val_comp_pot_df.to_csv(path_proccessed_data + "2_PCA_comps/val_pca_comp_potential.csv") #Save val PCA comp pots (10, 270)
test_comp_pot_df.to_csv(path_proccessed_data + "2_PCA_comps/test_pca_comp_potential.csv") #Save test PCA comp pots (10, 270)
#
train_comp_perm_us_df.to_csv(path_proccessed_data + "2_PCA_comps/train_pca_comp_permeability_unscaled.csv") #Save train Un-scaled PCA comp perm (282, 246) 
val_comp_perm_us_df.to_csv(path_proccessed_data + "2_PCA_comps/val_pca_comp_permeability_unscaled.csv") #Save val Un-scaled PCA comp perm (10, 246)
test_comp_perm_us_df.to_csv(path_proccessed_data + "2_PCA_comps/test_pca_comp_permeability_unscaled.csv") #Save test Un-scaled PCA comp perm (10, 246)

#*********************************************************************;
#  6. Plot PCA reconstruction errors of permeability (unscaled data)  ;
#*********************************************************************;
plot_gt_pred(train_perm_us.flatten(), train_perm_us_it.flatten(), \
	         str_x_label = 'Ground truth (Permeability)', \
	         str_y_label = 'Prediction (PCA Permeability)', \
	         str_fig_name = path_proccessed_data + "PCA_train_perm_error_unscaled") #(training perm) 282*585453
plot_gt_pred(val_perm_us.flatten(), val_perm_us_it.flatten(), \
	         str_x_label = 'Ground truth (Permeability)', \
	         str_y_label = 'Prediction (PCA Permeability)', \
	         str_fig_name = path_proccessed_data + "PCA_val_perm_error_unscaled") #(validation perm) 10*585453
plot_gt_pred(test_perm_us.flatten(), test_perm_us_it.flatten(), \
	         str_x_label = 'Ground truth (Permeability)', \
	         str_y_label = 'Prediction (PCA Permeability)', \
	         str_fig_name = path_proccessed_data + "PCA_test_perm_error_unscaled") #(testing perm) 10*585453

end_time   = time.time() #End timing
combo_time = end_time - start_time # Calculate total runtime
print("Total time = ", combo_time)