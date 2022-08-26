# Generate combined training and validation loss (Pre-mixup)
# Best set of 15 models -- for 15 different random seeds
# Inputs  -- encoded electrical potential difference measurements (Var = 0.99; n_comps = 270)
# Outputs -- encoded permeability field (Var = 0.99; n_comps = 246)
# Analysis is performed for Pre-Mixup --> m2_15randsd_302realz_mp.py
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

#================================================================================;
#  Function-1: Plot training and validation loss ('mse') for all 15 randomseeds  ;
#              (all random seeds, lines only)                                    ; 
#================================================================================;
def plot_tv_loss_allseeds(y_mat_data1, y_mat_data2, \
                          epochs, num_random_seeds, str_fig_name):

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
    ax.set_ylim([0, 2500])
    e_list = [i for i in range(0,epochs)]
    for i in range(0,num_random_seeds):
        if i == 0:
            ax.plot(e_list, y_mat_data1[:,i], linestyle = 'solid', linewidth = 0.75, \
                    color = 'b', alpha = 0.5, label = 'Training') #Training loss
            ax.plot(e_list, y_mat_data2[:,i], linestyle = 'solid', linewidth = 0.75, \
                    color = 'm', alpha = 0.5, label = 'Validation') #Validation loss
        else:
            ax.plot(e_list, y_mat_data1[:,i], linestyle = 'solid', linewidth = 0.75, \
                    color = 'b', alpha = 0.5) #Training loss
            ax.plot(e_list, y_mat_data2[:,i], linestyle = 'solid', linewidth = 0.75, \
                    color = 'm', alpha = 0.5) #Validation loss
    tick_spacing = 500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#================================================================================;
#  Function-2: Plot training and validation loss ('mse') for all 15 randomseeds  ;
#              (all random seeds, fill between only)                             ; 
#================================================================================;
def plot_tv_loss_fillbetween(y_mat_data1, y_mat_data2, y_lb1, y_lb2, y_ub1, y_ub2, \
                            epochs, str_fig_name):

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
    ax.plot(e_list, y_mat_data1[:,1], linestyle = 'solid', linewidth = 1.5, \
            color = 'b', alpha = 0.5, label = 'Training') #Training loss for random_seed = 0
    ax.plot(e_list, y_mat_data2[:,1], linestyle = 'solid', linewidth = 1.5, \
            color = 'm', alpha = 0.5, label = 'Validation') #Validation loss for random_seed = 0
    ax.fill_between(e_list, y_lb1, y_ub1, linestyle = 'solid', linewidth = 0.5, \
                    color = 'b', alpha = 0.2) #Mean +/ 2*std or 95% CI for Training loss
    ax.fill_between(e_list, y_lb1, y_ub1, linestyle = 'solid', linewidth = 0.5, \
                    color = 'm', alpha = 0.2) #Mean +/ 2*std or 95% CI for Validation loss
    tick_spacing = 500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = 'upper right')
    fig.tight_layout()
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#================================================================================;
#  Function-3: Plot training and validation loss ('mse') for all 15 randomseeds  ;
#              (all random seeds, box-plot only)                                 ; 
#================================================================================;
def plot_tv_loss_boxplot(y_mat_data, col_data, epochs, \
                        str_x_label, str_y_label, str_fig_name):

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
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([0, epochs])
    #ax.set_ylim([0.35, 1])
    e_list    = [str(i) for i in range(0,epochs)] #Boxplot labels
    data_list = [y_mat_data[i,:] for i in range(0,epochs)] #Boxplot data as a list
    ax.boxplot(data_list, vert = True, patch_artist = True) #Training loss for random_seed = 0
    tick_spacing = 500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fig.tight_layout()
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

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
    path_rand_seed_sav      = path + "302_Realz_Models/6_RandSeed_Predictions/pinklady_sims/2_m2_PreMixup/" #Save DL models (pinklady)
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

    #-----------------------------------------------;
    #  2. Model loss (MSE) training and validation  ;
    #-----------------------------------------------;
    epochs               = 500
    num_random_seeds     = 15 #Total number of random seeds to be used
    random_seed_list     = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                            11, 13, 42, 100, 113, 123, 200, \
                            1014, 1234, 1410, 1999, 12345, 31337, \
                            141079, 380843] #25 popular random seeds
    train_loss_mat       = np.ndarray(shape = (epochs,num_random_seeds), dtype = float) #(500,15)
    val_loss_mat         = np.ndarray(shape = (epochs,num_random_seeds), dtype = float) #(500,15)
    #
    for i in range(0,num_random_seeds):
        random_seed         = random_seed_list[i]
        path_fl_sav         = path_rand_seed_sav + str(random_seed) + "_model/"
        #
        train_csv           = path_fl_sav + "InvPCAModel2_training.csv"
        df_hist             = pd.read_csv(train_csv)
        train_loss_mat[:,i] = copy.deepcopy(df_hist['loss'].values)
        val_loss_mat[:,i]   = copy.deepcopy(df_hist['val_loss'].values)
        #
        print('i, random seed, = ', i, random_seed)

    df_train_loss_mat = pd.DataFrame(train_loss_mat, index = [str(i) for i in range(0,epochs)], \
                                    columns = [str(random_seed_list[i]) for i in range(0,num_random_seeds)]) #[500 rows x 15 columns]
    df_val_loss_mat   = pd.DataFrame(val_loss_mat, index = [str(i) for i in range(0,epochs)], \
                                    columns = [str(random_seed_list[i]) for i in range(0,num_random_seeds)]) #[500 rows x 15 columns]
    #
    train_loss_lb     = np.amin(train_loss_mat, axis = 1) #(500,)
    train_loss_ub     = np.amax(train_loss_mat, axis = 1) #(500,)
    val_loss_lb       = np.amin(val_loss_mat, axis = 1) #(500,)
    val_loss_ub       = np.amax(val_loss_mat, axis = 1) #(500,)

    #--------------------------------------;
    #  3. Plot train and val loss ('mse')  ;
    #     (all random seeds as lines)      ;
    #--------------------------------------;
    str_fig_name = path_rand_seed_sav + 'Loss_Train_Val_Allseeds' 
    plot_tv_loss_allseeds(train_loss_mat, val_loss_mat, \
                          epochs, num_random_seeds, str_fig_name)
    #
    str_fig_name = path_rand_seed_sav + 'Loss_Train_Val_Allseeds_FillBetween'     
    plot_tv_loss_fillbetween(train_loss_mat, val_loss_mat, \
                            train_loss_lb, train_loss_ub, \
                            val_loss_lb, val_loss_ub, \
                            epochs, str_fig_name)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)