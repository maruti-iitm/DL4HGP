# SCALABLE TRAINING ON PINKLADY -- Train 15 best models on 15 different processors (mpi4py)
#
# mpirun -n 15 python m1_15randsd_302realz_mp.py >> m1_15randsd_302realz_mp.txt
#
# DL training at scale -- Embarasingly parallel
#
# DL model training on 302 REALZ DATA (see m1_160comps_15randsd.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Encoded FC-DNNs (Strategy-m1: No Mixup) for 15 different random seeds
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

from mpi4py import MPI

#===============================================================================;
#  Function-1: Encoded potential-to-permeability for 15 different random seeds  ;
#===============================================================================;
def best_dnn_models(npotential_comps, nperm_comps, \
                    loop_units, alpha_units, \
                    dropout_units, dense_units, \
                    random_seed):

    #------------------------------------------------;
    #  Construct FC-DNN with different random seeds  ;
    #------------------------------------------------;
    pot_shape      = (npotential_comps,)
    #
    input_layer    = Input(shape = pot_shape, name = "PCA-pot-comps-in")
    x              = input_layer
    #  
    for i in range(loop_units):
        x = Dense(units = dense_units[i], name = "Dense-" + str(i))(x)
        x = LeakyReLU(alpha = alpha_units, name = "Activation-" + str(i))(x)
        x = Dropout(dropout_units)(x)

    out   = Dense(units = nperm_comps, name = "PCA-perm-comps-out")(x)
    model = Model(input_layer, out, name = "Inverse-PCA-Model-1-" + str(random_seed))

    return model

#=========================================================;
#  Function-2: Plot training and validation loss ('mse')  ; 
#=========================================================;
def plot_tv_loss(hist, epochs, path_fl_sav):

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
    fig.savefig(path_fl_sav + 'Loss_' + str(epochs) + '.pdf')
    fig.savefig(path_fl_sav + 'Loss_' + str(epochs) + '.png')

#=========================================================================;
#  Function-3: Plot one-to-one for train/val/test (for a given PCA comp)  ; 
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

#====================================================================;
#  Function-4: Train individual models (mpi4py calls this function)  ; 
#====================================================================;
def get_trained_models(realz_number):

    #-------------------;
    #  0. Get realz_id  ;
    #-------------------;
    i = realz_number - 1

    #------------------------------------------------;
    #  1. Get pre-processed data (all realizations)  ;
    #------------------------------------------------;
    #path = os.getcwd() #Get current directory path
    path                = "/home/mudu605/3_HGP_ML/m1_302realz/"
    path_rand_seed_sav  = path + "1_m1_NoMixup/" #Save DL models (No-mixup)
    #
    train_pots_fl       = path + "train_pca_comp_potential.csv" #Train inputs path
    val_pots_fl         = path + "val_pca_comp_potential.csv"  #Val inputs path
    test_pots_fl        = path + "test_pca_comp_potential.csv" #Test inputs path
    #
    train_perm_fl       = path + "train_pca_comp_permeability_unscaled.csv" #Train outputs path
    val_perm_fl         = path + "val_pca_comp_permeability_unscaled.csv" #Val outputs path
    test_perm_fl        = path + "test_pca_comp_permeability_unscaled.csv" #Test outputs path
    #
    perm_us_pca_name    = path + "perm_pca_unscaled_model.sav" #ln[K] fit on training data (Un-scaled)
    perm_us_pca_model   = pickle.load(open(perm_us_pca_name, 'rb')) #Load already created PCA ln[K] model (Unscaled)

    #------------------------------;
    #  2. Train/val/test PCA data  ;
    #------------------------------;
    train_index_list     = np.genfromtxt(path + "Train_Realz.txt", dtype = int, skip_header = 1) #Train indices  
    val_index_list       = np.genfromtxt(path + "Val_Realz.txt", dtype = int, skip_header = 1) #Val indices
    test_index_list      = np.genfromtxt(path + "Test_Realz.txt", dtype = int, skip_header = 1) #Test indices
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
    random_seed_list     = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                            11, 13, 42, 100, 113, 123, 200, \
                            1014, 1234, 1410, 1999, 12345, 31337, \
                            141079, 380843] #25 popular random seeds
    loop_units_list      = [1, 1, 1, 1, 1, \
                            1, 1, 1, 1, 1, \
                            5, 1, 2, 2, 1]
    alpha_units_list     = [0.5, 0.5, 0.5, 0.5, 0.45, \
                            0.5, 0.3, 0.5, 0.5, 0.5, \
                            0.5, 0.5, 0.5, 0.5, 0.5]
    dropout_units_list   = [0.5, 0.5, 0.5, 0.5, 0.5, \
                            0.5, 0.5, 0.5, 0.5, 0.5, \
                            0.0, 0.5, 0.35, 0.5, 0.5]
    dense_units_list     = [[300], [270], [300], [270], [270], \
                            [270], [270], [290], [300], [290], \
                            [300, 270, 270, 300, 270], [290], [280, 280], [280, 300], [270]]
    lr_values_list       = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, \
                            1e-3, 1e-3, 1e-3, 1e-3, 1e-3, \
                            1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    #
    random_seed   = random_seed_list[i]
    loop_units    = loop_units_list[i]
    alpha_units   = alpha_units_list[i]
    dropout_units = dropout_units_list[i] #Is a list
    dense_units   = dense_units_list[i]
    lr_values     = lr_values_list[i]
    #
    print('i, random seed, loop_units, alpha_units, dropout_units, dense_units, lr_values = ', \
            i, random_seed, loop_units, alpha_units, dropout_units, dense_units, lr_values)
    #
    K.clear_session()
    # 
    path_fl_sav   = path_rand_seed_sav + str(random_seed) + "_model/"
    #
    np.random.seed(random_seed)
    #
    model1 = best_dnn_models(npotential_comps, nperm_comps, \
                             loop_units, alpha_units, \
                             dropout_units, dense_units, \
                             random_seed) #Inverse PCA Model-1
    model1.summary()
    tf.keras.utils.plot_model(model1, path_fl_sav + "InvPCAModel-1a.png", show_shapes = True)
    tf.keras.utils.plot_model(model1, path_fl_sav + "InvPCAModel-1b.png", show_shapes = False)
    #
    opt        = Adam(learning_rate = lr_values)
    loss       = "mse"
    model1.compile(opt, loss = loss)
    epochs     = 4000
    train_csv  = path_fl_sav + "InvPCAModel1_training.csv"
    csv_logger = CSVLogger(train_csv)
    callbacks  = [csv_logger] 
    #
    history    = model1.fit(x = train_pots, y = train_perm, \
                            epochs = epochs, batch_size = 10, \
                            validation_data = (val_pots, val_perm), \
                            verbose = 2, callbacks = callbacks)
    hist = history.history
    print("Done training")
    #print(hist.keys())
    time.sleep(60)

    #--------------------------------------;
    #  4. Plot train and val loss ('mse')  ;
    #     (loss and epoch stats)           ;
    #--------------------------------------;
    #plot_tv_loss(hist, epochs, path_fl_sav)
    #
    df_hist        = pd.read_csv(train_csv)
    val_f1         = df_hist['val_loss']
    min_val_f1     = val_f1.min()
    min_val_f1_df  = df_hist[val_f1 == min_val_f1]
    min_epochs     = min_val_f1_df['epoch'].values
    min_val_loss   = min_val_f1_df['val_loss'].values
    min_train_loss = min_val_f1_df['loss'].values
    #
    print(min_val_f1_df)

    #----------------------------------------;
    #  5. Model prediction (train/val/test)  ;
    #----------------------------------------;
    train_pred_perm    = model1.predict(train_pots)
    val_pred_perm      = model1.predict(val_pots)
    test_pred_perm     = model1.predict(test_pots)
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
    #plot_gt_pred(x_train, y_train, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_val    = copy.deepcopy(val_perm.flatten())
    y_val    = copy.deepcopy(val_pred_perm.flatten())
    fl_name  = path_fl_sav + "P_val_"
    #plot_gt_pred(x_val, y_val, param_id, fl_name, \
    #                     str_x_label, str_y_label)
    #
    x_test   = copy.deepcopy(test_perm.flatten())
    y_test   = copy.deepcopy(test_pred_perm.flatten())
    fl_name  = path_fl_sav + "P_test_"
    #plot_gt_pred(x_test, y_test, param_id, fl_name, \
    #             str_x_label, str_y_label)

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
    #plot_gt_pred(x_train, y_train, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_val    = copy.deepcopy(val_perm_it.flatten())
    y_val    = copy.deepcopy(val_pred_perm_it.flatten())
    fl_name  = path_fl_sav + "IT_val_"
    #plot_gt_pred(x_val, y_val, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_test   = copy.deepcopy(test_perm_it.flatten())
    y_test   = copy.deepcopy(test_pred_perm_it.flatten())
    fl_name  = path_fl_sav + "IT_test_"
    #plot_gt_pred(x_test, y_test, param_id, fl_name, \
    #             str_x_label, str_y_label)

    #------------------------------------------------------------;
    #  9. PCA inverse transform the perm comps (train/val/test)  ;
    #------------------------------------------------------------;
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
    #plot_gt_pred(x_train, y_train, param_id, fl_name, \
    #            str_x_label, str_y_label)
    #

    x_val    = copy.deepcopy(val_perm_pcait.flatten())
    y_val    = copy.deepcopy(val_pred_perm_pcait.flatten())
    fl_name  = path_fl_sav + "Perm_val_pcait_"
    #plot_gt_pred(x_val, y_val, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_test   = copy.deepcopy(test_perm_pcait.flatten())
    y_test   = copy.deepcopy(test_pred_perm_pcait.flatten())
    fl_name  = path_fl_sav + "Perm_test_pcait_"
    #plot_gt_pred(x_test, y_test, param_id, fl_name, \
    #            str_x_label, str_y_label)

    #--------------------------------------------------------------;
    #  11. Load original ln[K] permeability data (train/val/test)  ;
    #      (For plotting purposes only)                            ;
    #--------------------------------------------------------------;
    perm_unscaled_data = np.load(path + "ln_Permeability_302_585453.npy") #Unscaled ln[K] data (302,585453)
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
    #plot_gt_pred(x_train, y_train, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_val    = copy.deepcopy(val_perm_us.flatten())
    y_val    = copy.deepcopy(val_pred_perm_pcait.flatten())
    fl_name  = path_fl_sav + "Perm_val_gt_"
    #plot_gt_pred(x_val, y_val, param_id, fl_name, \
    #             str_x_label, str_y_label)
    #
    x_test   = copy.deepcopy(test_perm_us.flatten())
    y_test   = copy.deepcopy(test_pred_perm_pcait.flatten())
    fl_name  = path_fl_sav + "Perm_test_gt_"
    #plot_gt_pred(x_test, y_test, param_id, fl_name, \
    #             str_x_label, str_y_label)

    #--------------------------------------------------------------------;
    #  13. Save ground truth and prediction ln[K] data (train/val/test)  ;
    #--------------------------------------------------------------------;
    np.save(path_fl_sav + "train_perm_us.npy", \
            train_perm_us) #Save train ground truth unscaled ln[K] data (282, 585453) in *.npy file
    np.save(path_fl_sav + "val_perm_us.npy", \
            val_perm_us) #Save val ground truth unscaled ln[K] data (10, 585453) in *.npy file
    np.save(path_fl_sav + "test_perm_us.npy", \
            test_perm_us) #Save test ground truth unscaled ln[K] data (10, 585453) in *.npy file
    #
    np.save(path_fl_sav + "train_perm_pred.npy", \
            train_pred_perm_pcait) #Save train predictions unscaled ln[K] data (282, 585453) in *.npy file
    np.save(path_fl_sav + "val_perm_pred.npy", \
            val_pred_perm_pcait) #Save val predictions unscaled ln[K] data (10, 585453) in *.npy file
    np.save(path_fl_sav + "test_perm_pred.npy", \
            test_pred_perm_pcait) #Save test predictions unscaled ln[K] data (10, 585453) in *.npy file

    #--------------------------------------------------------------;
    #  14. Save model (TensorFlow SavedModel format. *.h5 format)  ;
    #--------------------------------------------------------------;
    model1.save(path_fl_sav + "m1_NoMixup_v1") #TensorFlow SavedModel format
    model1.save(path_fl_sav + "m1_NoMixup_v1.h5") #h5 format

#********************************************************************;
#  Set paths, load preprocessed encoded PCA dataand dump .csv files  ;
#********************************************************************;
#
if __name__ == '__main__':

    #============================;
    #  1. Start processing time  ;
    #============================;
    tic = time.perf_counter()

    #=======================================;
    #  2. MPI communicator, size, and rank  ;
    #=======================================;
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #============================================;
    #  3. Number of realz per process/rank/core  ;
    #============================================;
    num_total = 1 #Total number of realization that needs to run on a given process/rank/core

    #=========================================;
    #  4. MPI send and receive realz numbers  ;
    #=========================================;
    if rank == 0:
        for i in range(size-1,-1,-1):
            realz_id = [0]*num_total  #Realization list
            #
            for j in range(num_total):
                print(j + num_total*i + 1, realz_id)
                realz_id[j] = j + num_total*i + 1
            print('rank and realz_id = ', rank, realz_id)
            #
            if i > 0: 
                comm.send(realz_id, dest = i)
    else:
        realz_id = comm.recv(source = 0)
        #print('rank, realz_id = ', rank, realz_id)

    #==========================================;
    #  5. Run DL model training for each realz ;
    #==========================================;
    for k in realz_id:
        get_trained_models(k)
        print('rank, k, realz_id = ', rank, k, realz_id)

    #=========================;
    #  6.End processing time  ;
    #=========================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic)