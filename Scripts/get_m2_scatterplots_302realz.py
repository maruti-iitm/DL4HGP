# Scatter plots for 302 REALZ DATA (see m2_160comps_15randsd.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Encoded FC-DNNs (Strategy-m2: Pre-Mixup) for 15 different random seeds
# Best set of 15 models
# Inputs  -- encoded electrical potential difference measurements (Var = 0.99; n_comps = 270)
# Outputs -- encoded permeability field (Var = 0.99; n_comps = 246)
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import time
import yaml
import pydot
import graphviz
import pydotplus
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

#=========================================================;
#  Function-1: Plot training and validation loss ('mse')  ; 
#=========================================================;
def plot_tv_loss(hist, epochs, normalization_name, path_fl_sav):

    #---------------------;
    #  Plot loss ('mse')  ;
    #---------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch', fontsize = 24, fontweight = 'bold')
    ax.set_ylabel('Loss (MSE)', fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([0, epochs])
    #ax.set_ylim([0.35, 1])
    e_list = [i for i in range(0,epochs)]
    sns.lineplot(e_list, hist['loss'], linestyle = 'solid', linewidth = 1.5, \
                 color = 'b', label = 'Training') #Training loss
    sns.lineplot(e_list, hist['val_loss'], linestyle = 'solid', linewidth = 1.5, \
                 color = 'm', label = 'Validation') #Validation loss
    tick_spacing = 500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(path_fl_sav + 'Loss_' + normalization_name + '.pdf')
    fig.savefig(path_fl_sav + 'Loss_' + normalization_name + '.png')

#=========================================================================;
#  Function-2: Plot one-to-one for train/val/test (for a given PCA comp)  ; 
#=========================================================================;
def plot_gt_pred(x, y, param_id, fl_name, \
                 str_x_label, str_y_label):

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
    #fig.savefig(fl_name + str(param_id) + '.pdf')
    fig.savefig(fl_name + str(param_id) + '.png')

#********************************************************************;
#  Set paths, load preprocessed encoded PCA dataand dump .csv files  ;
#********************************************************************;
#
if __name__ == '__main__':

    #=========================;
    #  Start processing time  ;
    #=========================;
    tic = time.perf_counter()

    #------------------------------------------------;
    #  1. Get pre-processed data (all realizations)  ;
    #------------------------------------------------;
    #path = os.getcwd() #Get current directory path
    path                    = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
    path_proccessed_data    = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
    path_ss_pca_sav         = path + "302_Realz_Models/1_SS_PCA_models/" #Saved StandardScalar and PCA models
    path_DNN_pred_sav       = path + "302_Realz_Models/3_DL_Predictions_Data/" #Saved DL models predictions of perm
    path_rand_seed_sav      = path + "302_Realz_Models/6_RandSeed_Predictions/pinklady_sims/2_m2_PreMixup/" #Save DL models
    #
    normalization_name_list = ["Not_Normalized"] #Three different datasets
    normalization_name      = normalization_name_list[0] #Choose the data to run (always use un-normalized PCA perm)
    #
    if normalization_name == "Not_Normalized": #PCA perm comps unnormalized
        train_pots_fl     = path_proccessed_data + "2_PCA_comps/train_pca_comp_potential.csv"
        val_pots_fl       = path_proccessed_data + "2_PCA_comps/val_pca_comp_potential.csv" 
        test_pots_fl      = path_proccessed_data + "2_PCA_comps/test_pca_comp_potential.csv"
        #
        train_perm_fl     = path_proccessed_data + "2_PCA_comps/train_pca_comp_permeability_unscaled.csv"
        val_perm_fl       = path_proccessed_data + "2_PCA_comps/val_pca_comp_permeability_unscaled.csv"
        test_perm_fl      = path_proccessed_data + "2_PCA_comps/test_pca_comp_permeability_unscaled.csv"
        #
        perm_us_pca_name  = path_ss_pca_sav + "1_PCA/perm_pca_unscaled_model.sav" #ln[K] fit on training data (Un-scaled)
        perm_us_pca_model = pickle.load(open(perm_us_pca_name, 'rb')) #Load already created PCA ln[K] model (Unscaled)
        #
        print("Data used is = ", normalization_name)

    #------------------------------;
    #  2. Train/val/test PCA data  ;
    #------------------------------;
    train_index_list     = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                         "Train_Realz.txt", dtype = int, skip_header = 1) #Train indices  
    val_index_list       = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                         "Val_Realz.txt", dtype = int, skip_header = 1) #Val indices
    test_index_list      = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                         "Test_Realz.txt", dtype = int, skip_header = 1) #Test indices
    #
    df_train_pots        = pd.read_csv(train_pots_fl, index_col = 0) #PCA train pot comps (282, 270)
    df_val_pots          = pd.read_csv(val_pots_fl, index_col = 0) #PCA val pot comps (10, 270)
    df_test_pots         = pd.read_csv(test_pots_fl, index_col = 0) #PCA test pot comps (10, 270)
    #
    df_train_perm        = pd.read_csv(train_perm_fl, index_col = 0) #PCA train perm comps (282, 246)
    df_val_perm          = pd.read_csv(val_perm_fl, index_col = 0) #PCA val perm comps (10, 246)
    df_test_perm         = pd.read_csv(test_perm_fl, index_col = 0) #PCA test perm comps (10, 246)
    #
    train_pots           = df_train_pots.values #Training PCA pots data (282, 270)
    val_pots             = df_val_pots.values #Validation PCA pots data (10, 270)
    test_pots            = df_test_pots.values #Test PCA pots data (10, 270)
    #
    train_perm           = df_train_perm.values #Training PCA perm data (282, 246)
    val_perm             = df_val_perm.values #Validation PCA perm data (10, 246)
    test_perm            = df_test_perm.values #Test PCA perm data (10, 246)
    #
    npotential_comps     = df_train_pots.shape[1] #No. of PCA pot comps #270
    nperm_comps          = df_train_perm.shape[1] #No. of PCA perm comps #246
    #
    pot_comps_cols_list  = df_train_pots.columns.to_list() #Pots PCA list -- A total of 270 comps 
    perm_comps_cols_list = df_train_perm.columns.to_list() #Perm PCA list -- A total of 246 comps

    #------------------------------------------------------------;
    #  3. Model training and validation                          ;
    #     Scenario-1: Batch size = 10; lr = 1e-4; epochs = 500   ;
    #     Scenario-2: Batch size = 10; lr = 1e-5; epochs = 4000  ;
    #------------------------------------------------------------;
    num_random_seeds     = 15 #Total number of random seeds to be used
    random_seed_list     = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                            11, 13, 42, 100, 113, 123, 200, \
                            1014, 1234, 1410, 1999, 12345, 31337, \
                            141079, 380843] #25 popular random seeds
    for i in range(0,num_random_seeds):
        random_seed   = random_seed_list[i]
        #
        path_fl_sav   = path_rand_seed_sav + str(random_seed) + "_model/"
        #
        print('i, random seed = ', i, random_seed)
        #
        np.random.seed(random_seed)

        #--------------------------------------;
        #  4. Plot train and val loss ('mse')  ;
        #     (loss and epoch stats)           ;
        #--------------------------------------;
        epochs         = 500
        train_csv      = path_fl_sav + "InvPCAModel2_training.csv"
        df_hist        = pd.read_csv(train_csv)
        val_f1         = df_hist['val_loss']
        min_val_f1     = val_f1.min()
        min_val_f1_df  = df_hist[val_f1 == min_val_f1]
        min_epochs     = min_val_f1_df['epoch'].values
        min_val_loss   = min_val_f1_df['val_loss'].values
        min_train_loss = min_val_f1_df['loss'].values
        #
        print(min_val_f1_df)
        plot_tv_loss(df_hist, epochs, normalization_name, path_fl_sav)

        #----------------------------------------;
        #  5. Model prediction (train/val/test)  ;
        #----------------------------------------;
        model2     = tf.keras.models.load_model(path_fl_sav + "m2_PreMixup_v1") #Load saved DNN model (encoded)
        model2.summary() #Print dnn model summary
        print(model2.layers[-1].output_shape[1:]) #Print output shape (no. of neurons in out-layer)
        #
        train_pred_perm    = model2.predict(train_pots)
        val_pred_perm      = model2.predict(val_pots)
        test_pred_perm     = model2.predict(test_pots)
        #
        df_train_pred_perm = pd.DataFrame(train_pred_perm, index=train_index_list, \
                                          columns = perm_comps_cols_list)
        df_val_pred_perm   = pd.DataFrame(val_pred_perm, index=val_index_list, \
                                          columns = perm_comps_cols_list)
        df_test_pred_perm  = pd.DataFrame(test_pred_perm, index=test_index_list, \
                                          columns = perm_comps_cols_list)

        #----------------------------------------------------;
        #  6. PCA perm comps un-normalized (train/val/test)  ;
        #----------------------------------------------------;
        if normalization_name == "MinMax_Scalar" or normalization_name == "Standard_Scalar":
            #
            print("TBD")
        else:
            train_pred_perm_it = copy.deepcopy(train_pred_perm) #Train perm pca comp preds (if un-normalized)
            val_pred_perm_it   = copy.deepcopy(val_pred_perm) #Val perm pca comp preds (if un-normalized)
            test_pred_perm_it  = copy.deepcopy(test_pred_perm) #Test perm pca comp preds (if un-normalized)
            #
            train_perm_it      = copy.deepcopy(train_perm) #Train perm pca comp ground truth (if un-normalized)
            val_perm_it        = copy.deepcopy(val_perm) #Val perm pca comp ground truth (if un-normalized)
            test_perm_it       = copy.deepcopy(test_perm) #Test perm pca comp ground truth (if un-normalized)

        #---------------------------------------------------;
        #  7. One-to-one normalized plots (train/val/test)  ;
        #---------------------------------------------------;
        str_x_label = 'Normalized ground truth'
        str_y_label = 'Normalized prediction'
        param_id    = "All_PCA_Comps"
        #
        x_train  = copy.deepcopy(train_perm.flatten())
        y_train  = copy.deepcopy(train_pred_perm.flatten())
        fl_name  = path_fl_sav + "P_train_"
        plot_gt_pred(x_train, y_train, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_val    = copy.deepcopy(val_perm.flatten())
        y_val    = copy.deepcopy(val_pred_perm.flatten())
        fl_name  = path_fl_sav + "P_val_"
        plot_gt_pred(x_val, y_val, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_test   = copy.deepcopy(test_perm.flatten())
        y_test   = copy.deepcopy(test_pred_perm.flatten())
        fl_name  = path_fl_sav + "P_test_"
        plot_gt_pred(x_test, y_test, param_id, fl_name, \
                         str_x_label, str_y_label)

        #----------------------------------------------------------;
        #  8. One-to-one inverse transform plots (train/val/test)  ;
        #----------------------------------------------------------;
        str_x_label = 'Ground truth (PCA comps)'
        str_y_label = 'Prediction (PCA comps)'
        param_id    = "All_PCA_Comps"
        #
        x_train  = copy.deepcopy(train_perm_it.flatten())
        y_train  = copy.deepcopy(train_pred_perm_it.flatten())
        fl_name  = path_fl_sav + "IT_train_"
        plot_gt_pred(x_train, y_train, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_val    = copy.deepcopy(val_perm_it.flatten())
        y_val    = copy.deepcopy(val_pred_perm_it.flatten())
        fl_name  = path_fl_sav + "IT_val_"
        plot_gt_pred(x_val, y_val, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_test   = copy.deepcopy(test_perm_it.flatten())
        y_test   = copy.deepcopy(test_pred_perm_it.flatten())
        fl_name  = path_fl_sav + "IT_test_"
        plot_gt_pred(x_test, y_test, param_id, fl_name, \
                         str_x_label, str_y_label)

        #------------------------------------------------------------;
        #  9. PCA inverse transform the perm comps (train/val/test)  ;
        #------------------------------------------------------------;
        if normalization_name == "MinMax_Scalar" or normalization_name == "Standard_Scalar":
           #
           print("TBD")
        else:
            train_pred_perm_pcait = perm_us_pca_model.inverse_transform(train_pred_perm_it) #PCA inverse transform predictions -- train (282, 585453)
            val_pred_perm_pcait   = perm_us_pca_model.inverse_transform(val_pred_perm_it) #PCA inverse transform predictions -- val (10, 585453)
            test_pred_perm_pcait  = perm_us_pca_model.inverse_transform(test_pred_perm_it) #PCA inverse transform predictions -- test (10, 585453)
            #
            train_perm_pcait      = perm_us_pca_model.inverse_transform(train_perm_it) #PCA inverse transform ground truth -- train (282, 585453)
            val_perm_pcait        = perm_us_pca_model.inverse_transform(val_perm_it) #PCA inverse transform ground truth -- val (10, 585453)
            test_perm_pcait       = perm_us_pca_model.inverse_transform(test_perm_it) #PCA inverse transform ground truth -- test (10, 585453)

        #---------------------------------------------------------------------------;
        #  10. One-to-one PCA inverse tranformed ln[K] plots (train/val/test)       ;
        #      (Predictions: PCA inverse transform of DNN estimate perm PCA comps)  ;
        #      (Ground truth: PCA inverse transform ground truth PCA comps)         ;
        #---------------------------------------------------------------------------;
        str_x_label = 'Ground truth (Permeability)'
        str_y_label = 'Prediction (Permeability)'
        param_id    = "All_Realz"
        #
        x_train  = copy.deepcopy(train_perm_pcait.flatten())
        y_train  = copy.deepcopy(train_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_train_pcait_"
        plot_gt_pred(x_train, y_train, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_val    = copy.deepcopy(val_perm_pcait.flatten())
        y_val    = copy.deepcopy(val_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_val_pcait_"
        plot_gt_pred(x_val, y_val, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_test   = copy.deepcopy(test_perm_pcait.flatten())
        y_test   = copy.deepcopy(test_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_test_pcait_"
        plot_gt_pred(x_test, y_test, param_id, fl_name, \
                         str_x_label, str_y_label)

        #--------------------------------------------------------------;
        #  11. Load original ln[K] permeability data (train/val/test)  ;
        #      (For plotting purposes only)                            ;
        #--------------------------------------------------------------;
        perm_unscaled_data = np.load(path_proccessed_data + \
                                     "0_Raw/ln_Permeability_302_585453.npy") #Unscaled ln[K] data (302,585453)
        #
        train_perm_us      = perm_unscaled_data[train_index_list-1,:] #Training unscaled ln[K] data (282, 585453)
        val_perm_us        = perm_unscaled_data[val_index_list-1,:] #Validation unscaled ln[K] data (10, 585453)
        test_perm_us       = perm_unscaled_data[test_index_list-1,:] #Test unscaled ln[K] data (10, 585453) 

        #---------------------------------------------------------------------------------;
        #  12. One-to-one PCA+DNN ln[K] vs. Original Ground truth plots (train/val/test)  ;
        #      (Predictions: PCA inverse transform of DNN estimate perm PCA comps)        ;
        #      (Ground truth: Original data)                                              ;
        #---------------------------------------------------------------------------------;    
        str_x_label = 'Ground truth (Permeability)'
        str_y_label = 'Prediction (Permeability)'
        param_id    = "All_Realz"
        #
        x_train  = copy.deepcopy(train_perm_us.flatten())
        y_train  = copy.deepcopy(train_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_train_gt_"
        plot_gt_pred(x_train, y_train, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_val    = copy.deepcopy(val_perm_us.flatten())
        y_val    = copy.deepcopy(val_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_val_gt_"
        plot_gt_pred(x_val, y_val, param_id, fl_name, \
                         str_x_label, str_y_label)
        #
        x_test   = copy.deepcopy(test_perm_us.flatten())
        y_test   = copy.deepcopy(test_pred_perm_pcait.flatten())
        fl_name  = path_fl_sav + "Perm_test_gt_"
        plot_gt_pred(x_test, y_test, param_id, fl_name, \
                         str_x_label, str_y_label)

    #======================;
    # End processing time  ;
    #======================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic) 