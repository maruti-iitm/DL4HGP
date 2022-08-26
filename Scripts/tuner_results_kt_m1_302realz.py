# RESULTS of KerasTuner -- DL model tuning on 302 REALZ DATA (see tuner_results_kt_m1_160comps.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Hyperparameter tuning for encoded DNNs (Strategy-m1: No Mixup)
# Inputs  -- encoded electrical potential difference measurements (Var = 0.99; n_comps = 270)
# Outputs -- encoded permeability field (Var = 0.99; n_comps = 246)
# AUTHOR: Maruti Kumar Mudunuru

import os
import pprint
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

import kerastuner as kt
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner import HyperModel

#====================================================================;
#  Class-1: Encoded potential-to-permeability (pot2perm) HyperModel  ;
#           (FC-DNN and No Mixup)                                    ;
#====================================================================;
class DNN_pot2perm_HyperModel(HyperModel):

    #-----------------;
    # Initialization  ;
    #-----------------;
    def __init__(self, npotential_comps, nperm_comps):
        self.npotential_comps   = npotential_comps #270
        self.nperm_comps        = nperm_comps #246

    #--------------;
    # Build model  ;
    #--------------;
    def build(self, hp):

        pot_shape      = (self.npotential_comps,)
        input_layer    = Input(shape = pot_shape)
        x              = input_layer
        #    
        loop_units     = hp.Int('num_layers', min_value = 1, max_value = 5, step = 1)
        alpha_values   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        dropout_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        #
        for i in range(loop_units):
            dense_units   = hp.Int(f"units_{i}",
                                   min_value = self.npotential_comps,
                                   max_value = 300,
                                   step = 10)
            x             = Dense(units = dense_units)(x)

            alpha_units   = hp.Choice('alpha', values = alpha_values)
            x             = LeakyReLU(alpha_units)(x)

            dropout_units = hp.Choice('rate', values = dropout_values)
            x             = Dropout(dropout_units)(x)

        out   = Dense(self.nperm_comps)(x)
        model = Model(input_layer, out)

        #Set optimizer
        lr_values = [1e-7, 5e-7, 1e-6, 2e-6, 4e-6, 6e-6, 8e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1e-3]
        opt_units = hp.Choice('learning_rate', values = lr_values)
        optimizer = tf.keras.optimizers.Adam(opt_units)
      
        #Set Loss
        loss    = "mse"
        metrics =['mean_squared_error']
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        return model

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
    path_rand_seed_sav      = path + "302_Realz_Models/6_RandSeed_Predictions/1_m1_NoMixup/" #Save DL models
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
    #  3. Hyperparameter tuning setup and associated parameters  ;
    #------------------------------------------------------------;
    num_random_seeds     = 15 #Total number of random seeds to be used
    random_seed_list     = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                            11, 13, 42, 100, 113, 123, 200, \
                            1014, 1234, 1410, 1999, 12345, 31337, \
                            141079, 380843] #25 popular random seeds
    i = 1
    #
    K.clear_session()
    print('i, random seed = ', i, random_seed_list[i])
    #
    tuner_type           = "bayesian_opt"
    objective            = "val_mean_squared_error" #Name of model metric to minimize or maximize, e.g. "val_accuracy".
    max_trials           = 40 #Total number of trials (model configurations) to test at most
    executions_per_trial = 1 #Number of models that should be built and fit for each trial
    epochs               = 4000 #Number of epochs
    project_name         = 'tuner_dnn' #Name to use as prefix for files saved by this Tuner.
    random_seed          = random_seed_list[i] #i-th random-seed
    verbose              = 2 #Detailed messages (e.g., epochs for train and val loss)
    num_batches          = 10 #Total number of batches
    #
    dnn_p2p_hypermodel   = DNN_pot2perm_HyperModel(npotential_comps = npotential_comps, nperm_comps = nperm_comps)
    directory            = path + "Python_Scripts/302_DL_Scripts/kt_m1_15randsd/" + \
                           str(random_seed) + "_" + str(num_batches) + "_" + str(epochs) + \
                           f"_tuner_{tuner_type}" #Path to the working directory (absolute).
    #directory            = f"tuner_{tuner_type}" #Path to the working directory (relative).
    #
    if tuner_type == 'randomsearch':
        tuner = RandomSearch(dnn_p2p_hypermodel,
                             objective = objective,
                             max_trials = max_trials,
                             executions_per_trial = executions_per_trial,
                             directory = directory,
                             project_name = project_name,
                             seed = random_seed)
    elif tuner_type == 'bayesian_opt':
        #Default values
        tune_new_entries   = True #True
        allow_new_entries  = True  #True
        num_initial_points = 20 #True
        #
        tuner = BayesianOptimization(dnn_p2p_hypermodel,
                                     objective = objective,
                                     max_trials = max_trials,
                                     executions_per_trial = executions_per_trial,
                                     directory = directory,
                                     project_name = project_name,
                                     seed = random_seed,
                                     num_initial_points = num_initial_points,
                                     tune_new_entries = tune_new_entries,
                                     allow_new_entries = allow_new_entries)  
    else:
        print(f"Incorrect tuner type: {tuner_type}")

    print(f"Tuner: {tuner_type}")
    tuner.search_space_summary() #Search space of the tuner
    tuner.reload() #Do this in a seperate file, when search is complete (remove the code below this) https://github.com/keras-team/keras-tuner/issues/41

    #-------------------------------------------------------------;
    #  4. Hyperparameter tuning -- Searching the parameter space  ;
    #-------------------------------------------------------------; 
    print(tuner.results_summary(num_trials = 15)) #Get first five best models
    
    #----------------------------------------;
    #  5. Best set of hyperparameter values  ;
    #----------------------------------------; 
    best_hp        = tuner.get_best_hyperparameters(num_trials = 5)[0] #Get first best model
    #best_hp        = tuner.get_best_hyperparameters(num_trials = 5)[1] #Get second best model and so on ...
    best_hp_values = best_hp.values
    #
    print(f'----------BEST MODEL for random seed:{random_seed}----------')
    print('i, random seed = ', i, random_seed)
    pprint.pprint(best_hp_values)
    print('-------------------------------------------------------------')
    #
    best_model = tuner.hypermodel.build(best_hp) #reinstantiate the (untrained) best models
    best_model.summary()
    print('\n')

    #======================;
    # End processing time  ;
    #======================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic)