# Metric error plots for 302 REALZ DATA (see get_errors_m1_160_15randsd.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Generate percent errors and 2D slices of the permeability field (Ground truth vs. predictions)
# Best set of 15 models -- for 15 different random seeds (Var = 0.99; n_comps = 246)
# Analysis is performed for No-Mixup --> m1_160comps_15randsd.py
# AUTHOR: Maruti Kumar Mudunuru
# Sample from a distribution: https://stackoverflow.com/questions/24364770/random-sampling-from-a-list-based-on-a-distribution

import os
import time
import copy
import pickle
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
#
from sklearn.metrics import mean_squared_error, \
                            r2_score, \
                            mean_absolute_error, \
                            explained_variance_score, \
                            max_error, \
                            mean_absolute_percentage_error, \
                            median_absolute_error
from skimage.metrics import structural_similarity as ssim_2d

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#===================================;
#  Function-1: Ticklabels formater  ;
#===================================;
def xaxis_format_func(value, tick_number):

    """Double xaxis value"""
    return(int(value)*2)

#============================================================;
#  Function-2: Index value to depth converter (90m to 110m)  ;
#============================================================;
def depth_converter(value):

    depth = 0.5*int(value) + 90

    return depth

#================================================;
#  Function-3: Plot 2D ln[K] permeability field  ;
#================================================;
def plot_horizontal_slice(perm_field, depth, str_fig_name, \
	                      xticklabels, yticklabels, vmin, vmax, \
	                      realz_indx, cmap):

    #------------------------------------------------------;
    #  Heatmap of 2D permeability field at various depths  ;
    #------------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    depth_m  = depth_converter(depth) #90m to 110m
    ho_slice = perm_field #2D slice
    ho_slice = ho_slice.transpose() #Transpose for correct orientation
    #
    ax       = sns.heatmap(ho_slice, xticklabels=xticklabels, yticklabels=yticklabels, \
    	                   cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, square=True)
    ax.invert_yaxis()
    #
    ax.set_title("Realz-{}".format(realz_indx) + "  Depth {}m".format(depth_m), fontsize = 24)
    ax.set_xlabel("Easting (m)", fontsize = 24, fontweight = 'bold')
    ax.set_ylabel("Northing (m)", fontsize = 24, fontweight = 'bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    #
    ax.xaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    #
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize = 24)
    fig.tight_layout()
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name +  str(depth) + '.png')

#========================================================;
#  Function-4: Plot lump distribution of percent errors  ;
#              (Histogram vs KDE)                        ;
#========================================================;
def plot_hist_kde(error_data, num_bins, x_kde, y_kde, xmin, xmax, ymin, ymax, \
    hist_label, kde_label, loc_pos, str_x_label, str_y_label, str_fig_name):

    #------------------------------------------------------;
    #  Heatmap of 2D permeability field at various depths  ;
    #------------------------------------------------------;
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
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    #
    ax.hist(error_data.flatten(), bins = num_bins, density = True, \
            label = hist_label, edgecolor = 'k', alpha = 0.5)
    #ax.plot(x_kde, y_kde, linewidth = 2.5, color = 'b', label = kde_label)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    #fig.savefig(str_title + '.pdf')
    fig.savefig(str_fig_name + '.png')

#========================================================;
#  Function-5: Plot lump distribution of percent errors  ;
#              (Histogram vs KDE)                        ;
#========================================================;
def plot_hist_kde_trainvaltest(x_data, y_data, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, width_val, str_x_label, str_y_label, str_fig_name):

    #------------------------------------------------------;
    #  Heatmap of 2D permeability field at various depths  ;
    #------------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 20)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #
    #ax.hist(error_data.flatten(), bins = num_bins, density = True, \
    #        label = hist_label, edgecolor = 'k', alpha = 0.5)
    ax.bar(x_data[1:], y_data, align = 'center', width = width_val, \
            label = hist_label, edgecolor = 'k', alpha = 0.5)
    #ax.plot(x_kde, y_kde, linewidth = 2.5, color = 'b', label = kde_label)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    #fig.savefig(str_title + '.pdf')
    fig.savefig(str_fig_name + '.png')

#=============================================================;
#  Function-6: Plot performance metrics (r2-score, ... etc )  ;
#=============================================================;
def plot_metrics_realz(x_data_list, y_data_list, loc_pos, \
                        xmin, xmax, ymin, ymax, num_data, \
                        color_list, marker_list, \
                        str_x_label, str_y_label, str_fig_name):

    #---------------------------------------------;
    #  Plot Performance metrics vs. realizations  ;
    #---------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 8)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #
    for i in range(0,num_data):
        ax.plot(x_data_list, y_data_list[:,i], linestyle = 'solid', linewidth = 1.0,\
                marker = marker_list[i], color = color_list[i], alpha = 0.5) #Metric for 15 random seed models
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.legend(loc = loc_pos)
    fig.tight_layout()
    #fig.savefig(str_title + '.pdf')
    fig.savefig(str_fig_name + '.png')

#==============================================;
#  Function-7: Plot lump distribution of KDEs  ;
#==============================================;
def plot_kde_trainvaltest(x_kde, y_kde1, y_kde2, y_kde3, \
    xmin, xmax, ymin, ymax, loc_pos, str_x_label, str_y_label, str_fig_name):

    #------------------------------------------------------;
    #  Heatmap of 2D permeability field at various depths  ;
    #------------------------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 20)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    #
    ax.plot(x_kde, y_kde1, linewidth = 1.5, color = 'b', label = 'Training set')
    ax.plot(x_kde, y_kde2, linewidth = 1.5, color = 'g', label = 'Validation set')
    ax.plot(x_kde, y_kde3, linewidth = 1.5, color = 'r', label = 'Testing set')
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#******************************************************************;
# 0. Set OLD data paths (ln[K] ground truth and prediction *.npy)  ; 
#******************************************************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
path_DNN_pred_sav    = path + "302_Realz_Models/3_DL_Predictions_Data/" #Saved DL models predictions of perm
path_rand_seed_sav   = path + "302_Realz_Models/6_RandSeed_Predictions/pinklady_sims/1_m1_NoMixup/" #Save DL models
path_newmask_id_npy  = path + "302_Realz_Models/New_Material_IDs_Plume_2mx2mxhalfm.npy" #New mask *.npy file (IDs = 0,1,4, and -1)

#********************************************;
# 1. Train/Val/Test data and initialization  ; 
#********************************************;
train_index_list = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                 "Train_Realz.txt", dtype = int, skip_header = 1) #Train indices  
val_index_list   = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                 "Val_Realz.txt", dtype = int, skip_header = 1) #Val indices
test_index_list  = np.genfromtxt(path_proccessed_data + "Train_Val_Test_Indices/" + \
                                 "Test_Realz.txt", dtype = int, skip_header = 1) #Test indices
#
train_perm       = np.load(path_DNN_pred_sav + "0_Raw_TrainValTest/train_perm_us.npy") #Training unscaled ln[K] GT data (180, 585453)
val_perm         = np.load(path_DNN_pred_sav + "0_Raw_TrainValTest/val_perm_us.npy") #Validation unscaled ln[K] GT data (10, 585453)
test_perm        = np.load(path_DNN_pred_sav + "0_Raw_TrainValTest/test_perm_us.npy") #Test unscaled ln[K] GT data (10, 585453) 
#
num_train        = 282 #282
num_val          = 10 #10
num_test         = 10 #10

#*******************************;
# 2. Masked cells information   ; 
#*******************************;
new_mask_id_list       = np.load(path_newmask_id_npy) #We have Cell IDs = 0, 1 (Hanford), 4, and -1 (107m to 109.5m)
num_cells              = new_mask_id_list.shape[0] #(1600000,)
#
new_nan_cells          = new_mask_id_list == 0 #(1600000,) -- River cells (Cell ID = 0)
new_tiny_cells         = new_mask_id_list == 4 #(1600000,) -- Ringold cells (Cell ID = 4)
new_z_cells            = new_mask_id_list == -1 #(1600000,) -- z = 107m to 109.5m (Cell ID = -1)
new_hf_cells           = new_mask_id_list == 1 #(1600000,) -- Hanford cells without z = 107m to 109.5m
#
new_nan_cells_id       = np.argwhere(new_mask_id_list == 0)[:,0] #(32397,) River cells IDs
new_tiny_cells_id      = np.argwhere(new_mask_id_list == 4)[:,0] #(742150,) Ringold cells IDs
new_z_cells_id         = np.argwhere(new_mask_id_list == -1)[:,0] #(240000,) z = 107m to 109.5m cells IDs
new_hf_cells_id        = np.argwhere(new_mask_id_list == 1)[:,0] #(585453,) Hanford cells IDs
num_new_mask_cells     = new_nan_cells_id.shape[0] + new_tiny_cells_id.shape[0] + \
                         new_z_cells_id.shape[0] #1014547 masked cells
num_new_non_mask_cells = num_cells - num_new_mask_cells #585453, which are all the Hanford cells (no 107m to 109.5m)
#
print('new_nan_cells_id = ', new_nan_cells_id, new_nan_cells_id.shape)
print('new_tiny_cells_id = ', new_tiny_cells_id, new_tiny_cells_id.shape)
print('new_z_cells_id = ', new_z_cells_id, new_z_cells_id.shape)
print('new_hf_cells_id = ', new_hf_cells_id, new_hf_cells_id.shape)
print('num_new_mask_cells, new_non-mask-cells = ', num_new_mask_cells, num_new_non_mask_cells)

#*******************************;
# 3. Loop over 15 random seeds  ; 
#*******************************;
num_random_seeds = 15 #Total number of random seeds to be used
random_seed_list = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                    11, 13, 42, 100, 113, 123, 200, \
                    1014, 1234, 1410, 1999, 12345, 31337, \
                    141079, 380843] #25 popular random seeds
#
r2_train_list    = np.zeros((num_train,num_random_seeds), dtype = float) #R2-score list for train
mse_train_list   = np.zeros((num_train,num_random_seeds), dtype = float) #mse list for train
rmse_train_list  = np.zeros((num_train,num_random_seeds), dtype = float) #rmse list for train
mae_train_list   = np.zeros((num_train,num_random_seeds), dtype = float) #mean_absolute_error list for train
evs_train_list   = np.zeros((num_train,num_random_seeds), dtype = float) #explained_variance_score list for train
me_train_list    = np.zeros((num_train,num_random_seeds), dtype = float) #max_error list for train
mape_train_list  = np.zeros((num_train,num_random_seeds), dtype = float) #mean_absolute_percentage_error list for train
medae_train_list = np.zeros((num_train,num_random_seeds), dtype = float) #median_absolute_error list for train
ssim_train_list  = np.zeros((num_train,num_random_seeds), dtype = float) #ssim list for train
pe_train_list    = np.zeros((num_train,num_random_seeds), dtype = float) #max percent error list for train
#
r2_val_list      = np.zeros((num_val,num_random_seeds), dtype = float) #R2-score list for val
mse_val_list     = np.zeros((num_val,num_random_seeds), dtype = float) #mse list for val
rmse_val_list    = np.zeros((num_val,num_random_seeds), dtype = float) #rmse list for val
mae_val_list     = np.zeros((num_val,num_random_seeds), dtype = float) #mean_absolute_error list for val
evs_val_list     = np.zeros((num_val,num_random_seeds), dtype = float) #explained_variance_score list for val
me_val_list      = np.zeros((num_val,num_random_seeds), dtype = float) #max_error list for val
mape_val_list    = np.zeros((num_val,num_random_seeds), dtype = float) #mean_absolute_percentage_error list for val
medae_val_list   = np.zeros((num_val,num_random_seeds), dtype = float) #median_absolute_error list for val
ssim_val_list    = np.zeros((num_val,num_random_seeds), dtype = float) #ssim list for val
pe_val_list      = np.zeros((num_val,num_random_seeds), dtype = float) #max percent error list for test
#
r2_test_list     = np.zeros((num_test,num_random_seeds), dtype = float) #R2-score list for test
mse_test_list    = np.zeros((num_test,num_random_seeds), dtype = float) #mse list for test
rmse_test_list   = np.zeros((num_test,num_random_seeds), dtype = float) #rmse list for test
mae_test_list    = np.zeros((num_test,num_random_seeds), dtype = float) #mean_absolute_error list for test
evs_test_list    = np.zeros((num_test,num_random_seeds), dtype = float) #explained_variance_score list for test
me_test_list     = np.zeros((num_test,num_random_seeds), dtype = float) #max_error list for test
mape_test_list   = np.zeros((num_test,num_random_seeds), dtype = float) #mean_absolute_percentage_error list for test
medae_test_list  = np.zeros((num_test,num_random_seeds), dtype = float) #median_absolute_error list for test
ssim_test_list   = np.zeros((num_test,num_random_seeds), dtype = float) #ssim list for test
pe_test_list     = np.zeros((num_test,num_random_seeds), dtype = float) #max percent error list for test
#
train_err_list   = np.zeros((num_train*num_random_seeds,new_hf_cells_id.shape[0]), dtype = float) #Percent error list for train
val_err_list     = np.zeros((num_val*num_random_seeds,new_hf_cells_id.shape[0]), dtype = float) #Percent error list for val
test_err_list    = np.zeros((num_test*num_random_seeds,new_hf_cells_id.shape[0]), dtype = float) #Percent error list for test
#
stride_train = 0
stride_val   = 0
stride_test  = 0
#
for i in range(0,num_random_seeds): 
    random_seed     = random_seed_list[i]
    path_fl_sav     = path_rand_seed_sav + str(random_seed) + "_model/"
    #
    train_perm_pred = np.load(path_fl_sav + "train_perm_pred.npy") #Training unscaled ln[K] pred data (180, 585453)
    val_perm_pred   = np.load(path_fl_sav + "val_perm_pred.npy") #Validation unscaled ln[K] pred data (10, 585453)
    test_perm_pred  = np.load(path_fl_sav + "test_perm_pred.npy") #Test unscaled ln[K] pred data (10, 585453)
    #
    print('i, random seed = ', i, random_seed)

    #******************************;
    # 4. Test plots (predictions)  ;
    #******************************;
    xticklabels   = 20
    yticklabels   = 20
    vmin          = 0.0 #ln[K] min value
    vmax          = 10.0 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    #
    for counter in list(range(0,num_test)):
        temp_pred                    = copy.deepcopy(test_perm_pred[counter,0:585225]) #for SSIM
        temp_true                    = copy.deepcopy(test_perm[counter,0:585225]) #for SSIM
        #
        temp_pred_2d                 = temp_pred.reshape((765,765)) #for SSIM
        temp_true_2d                 = temp_true.reshape((765,765)) #for SSIM
        #
        test_percent_error           = np.abs(np.divide(test_perm[counter,:] - test_perm_pred[counter,:], \
                                                test_perm[counter,:]))*100 #percent error
        test_err_list[stride_test,:] = copy.deepcopy(test_percent_error) #Build percent error list (150, 585453)
        #
        stride_test                  = stride_test + 1
        #
        r2           = np.abs(r2_score(test_perm[counter,:], test_perm_pred[counter,:])) #r2-score
        mse          = mean_squared_error(test_perm[counter,:], test_perm_pred[counter,:]) #MSE-score
        rmse         = mean_squared_error(test_perm[counter,:], test_perm_pred[counter,:], \
                                         squared = False) #RMSE-score
        mae          = mean_absolute_error(test_perm[counter,:], test_perm_pred[counter,:]) #mean_absolute_error
        evs          = np.abs(explained_variance_score(test_perm[counter,:], test_perm_pred[counter,:])) #explained_variance_score
        me           = max_error(test_perm[counter,:], test_perm_pred[counter,:]) #max_error
        mape         = mean_absolute_percentage_error(test_perm[counter,:], test_perm_pred[counter,:]) #mean_absolute_percentage_error
        medae        = median_absolute_error(test_perm[counter,:], test_perm_pred[counter,:]) #median_absolute_error 
        ssim         = ssim_2d(temp_true_2d, temp_pred_2d, win_size = 7, multichannel = False, K1 = 0.01, K2 = 0.03) #SSIM
        pe           = np.max(test_percent_error) #Max percent error
        #
        r2_test_list[counter,i]     = r2 #R2-score list for test
        mse_test_list[counter,i]    = mse #mse list for test
        rmse_test_list[counter,i]   = rmse #rmse list for test
        mae_test_list[counter,i]    = mae #mean_absolute_error list for test
        evs_test_list[counter,i]    = evs #explained_variance_score list for test
        me_test_list[counter,i]     = me #max_error list for test
        mape_test_list[counter,i]   = mape #mean_absolute_percentage_error list for test
        medae_test_list[counter,i]  = medae #median_absolute_error list for test
        ssim_test_list[counter,i]   = ssim #ssim list for test
        pe_test_list[counter,i]     = pe #max percent error list for test
        #
        realz_indx    = test_index_list[counter] #Test index
        str_fig_name  = path_fl_sav + "Test_Percent_Errors/" + str(realz_indx) + "_test_err_"
        perm_flat_err = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat_err[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat_err[new_tiny_cells_id] = 0.0 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat_err[new_z_cells_id]    = 0.0 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat_err[new_hf_cells_id]   = copy.deepcopy(test_percent_error) #(585453,) New Hanford cells IDs
        #
        perm_field_err                   = perm_flat_err.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        #for depth_id in depth_id_list:
        #    #print("Test counter, realz, depth = ", counter, realz_indx, depth_id)
        #    plot_horizontal_slice(perm_field_err[:, :, depth_id], depth_id, str_fig_name, \
        #                          xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)
        #print("i, Test = ", i, counter, "{:.2f}".format(r2), \
        #    "{:.2f}".format(mse), "{:.2f}".format(rmse), \
        #    "{:.2f}".format(mae), "{:.2f}".format(evs), \
        #    "{:.2f}".format(me), "{:.2f}".format(mape), \
        #    "{:.2f}".format(medae), "{:.2f}".format(ssim), \
        #    "{:.2f}".format(pe))

    #************************************;
    # 5. Validation plots (predictions)  ;
    #************************************;
    xticklabels   = 20
    yticklabels   = 20
    vmin          = 0.0 #ln[K] min value
    vmax          = 10.0 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    #
    for counter in list(range(0,num_val)):
        temp_pred                   = copy.deepcopy(val_perm_pred[counter,0:585225]) #for SSIM
        temp_true                   = copy.deepcopy(val_perm[counter,0:585225]) #for SSIM
        #
        temp_pred_2d                = temp_pred.reshape((765,765)) #for SSIM
        temp_true_2d                = temp_true.reshape((765,765)) #for SSIM
        #
        val_percent_error           = np.abs(np.divide(val_perm[counter,:] - val_perm_pred[counter,:], \
                                                val_perm[counter,:]))*100 #percent error
        val_err_list[stride_val,:]  = copy.deepcopy(val_percent_error) #Build percent error list (150, 585453)
        #
        stride_val                  = stride_val + 1
        #
        r2           = np.abs(r2_score(val_perm[counter,:], val_perm_pred[counter,:])) #r2-score
        mse          = mean_squared_error(val_perm[counter,:], val_perm_pred[counter,:]) #MSE-score
        rmse         = mean_squared_error(val_perm[counter,:], val_perm_pred[counter,:], \
                                         squared = False) #RMSE-score
        mae          = mean_absolute_error(val_perm[counter,:], val_perm_pred[counter,:]) #mean_absolute_error
        evs          = np.abs(explained_variance_score(val_perm[counter,:], val_perm_pred[counter,:])) #explained_variance_score
        me           = max_error(val_perm[counter,:], val_perm_pred[counter,:]) #max_error
        mape         = mean_absolute_percentage_error(val_perm[counter,:], val_perm_pred[counter,:]) #mean_absolute_percentage_error
        medae        = median_absolute_error(val_perm[counter,:], val_perm_pred[counter,:]) #median_absolute_error 
        ssim         = ssim_2d(temp_true_2d, temp_pred_2d, win_size = 7, multichannel = False, K1 = 0.01, K2 = 0.03) #SSIM
        pe           = np.max(val_percent_error) #Max percent error
        #
        r2_val_list[counter,i]     = r2 #R2-score list for val
        mse_val_list[counter,i]    = mse #mse list for val
        rmse_val_list[counter,i]   = rmse #rmse list for val
        mae_val_list[counter,i]    = mae #mean_absolute_error list for val
        evs_val_list[counter,i]    = evs #explained_variance_score list for val
        me_val_list[counter,i]     = me #max_error list for val
        mape_val_list[counter,i]   = mape #mean_absolute_percentage_error list for val
        medae_val_list[counter,i]  = medae #median_absolute_error list for val
        ssim_val_list[counter,i]   = ssim #ssim list for val
        pe_val_list[counter,i]     = pe #max percent error list for val
        #
        realz_indx    = val_index_list[counter] #Validation index
        str_fig_name  = path_fl_sav + "Val_Percent_Errors/" + str(realz_indx) + "_val_err_"
        perm_flat_err = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat_err[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat_err[new_tiny_cells_id] = 0.0 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat_err[new_z_cells_id]    = 0.0 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat_err[new_hf_cells_id]   = copy.deepcopy(val_percent_error) #(585453,) New Hanford cells IDs
        #
        perm_field_err                   = perm_flat_err.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        #for depth_id in depth_id_list:
        #    #print("Val counter, realz, depth = ", counter, realz_indx, depth_id)
        #    plot_horizontal_slice(perm_field_err[:, :, depth_id], depth_id, str_fig_name, \
	    #                          xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)
        #print("i, Val = ", i, counter, "{:.2f}".format(r2), \
        #    "{:.2f}".format(mse), "{:.2f}".format(rmse), \
        #    "{:.2f}".format(mae), "{:.2f}".format(evs), \
        #    "{:.2f}".format(me), "{:.2f}".format(mape), \
        #    "{:.2f}".format(medae), "{:.2f}".format(ssim), \
        #    "{:.2f}".format(pe))

    #**********************************;
    # 6. Training plots (predictions)  ;
    #**********************************;
    xticklabels   = 20
    yticklabels   = 20
    vmin          = 0.0 #ln[K] min value
    vmax          = 10.0 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    #counter_list  = [0, 152, 10, 101] #np.argwhere(train_index_list == 1)
    #
    for counter in list(range(0,num_train)):
    #for counter in counter_list:
        temp_pred                       = copy.deepcopy(train_perm_pred[counter,0:585225]) #for SSIM
        temp_true                       = copy.deepcopy(train_perm[counter,0:585225]) #for SSIM
        #
        temp_pred_2d                    = temp_pred.reshape((765,765)) #for SSIM
        temp_true_2d                    = temp_true.reshape((765,765)) #for SSIM
        #
        train_percent_error             = np.abs(np.divide(train_perm[counter,:] - train_perm_pred[counter,:], \
                                                train_perm[counter,:]))*100 #percent error
        train_err_list[stride_train,:]  = copy.deepcopy(train_percent_error) #Build percent error list (150, 585453)
        #
        stride_train                    = stride_train + 1
        #
        r2           = np.abs(r2_score(train_perm[counter,:], train_perm_pred[counter,:])) #r2-score
        mse          = mean_squared_error(train_perm[counter,:], train_perm_pred[counter,:]) #MSE-score
        rmse         = mean_squared_error(train_perm[counter,:], train_perm_pred[counter,:], \
                                         squared = False) #RMSE-score
        mae          = mean_absolute_error(train_perm[counter,:], train_perm_pred[counter,:]) #mean_absolute_error
        evs          = np.abs(explained_variance_score(train_perm[counter,:], train_perm_pred[counter,:])) #explained_variance_score
        me           = max_error(train_perm[counter,:], train_perm_pred[counter,:]) #max_error
        mape         = mean_absolute_percentage_error(train_perm[counter,:], train_perm_pred[counter,:]) #mean_absolute_percentage_error
        medae        = median_absolute_error(train_perm[counter,:], train_perm_pred[counter,:]) #median_absolute_error 
        ssim         = ssim_2d(temp_true_2d[counter,:], temp_pred_2d[counter,:], \
                                win_size = 7, multichannel = False, K1 = 0.01, K2 = 0.03) #SSIM
        pe           = np.max(train_percent_error) #Max percent error
        #
        r2_train_list[counter,i]     = r2 #R2-score list for val
        mse_train_list[counter,i]    = mse #mse list for val
        rmse_train_list[counter,i]   = rmse #rmse list for val
        mae_train_list[counter,i]    = mae #mean_absolute_error list for val
        evs_train_list[counter,i]    = evs #explained_variance_score list for val
        me_train_list[counter,i]     = me #max_error list for val
        mape_train_list[counter,i]   = mape #mean_absolute_percentage_error list for val
        medae_train_list[counter,i]  = medae #median_absolute_error list for val
        ssim_train_list[counter,i]   = ssim #ssim list for val
        pe_train_list[counter,i]     = pe #max percent error list for val
        #
        realz_indx    = train_index_list[counter] #Training index
        str_fig_name  = path_fl_sav + "Train_Percent_Errors/" + str(realz_indx) + "_train_err_"
        perm_flat_err = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat_err[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat_err[new_tiny_cells_id] = 0.0 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat_err[new_z_cells_id]    = 0.0 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat_err[new_hf_cells_id]   = copy.deepcopy(train_percent_error) #(585453,) New Hanford cells IDs
        #
        perm_field_err                   = perm_flat_err.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        #for depth_id in depth_id_list:
        #    #print("Train counter, realz, depth = ", counter, realz_indx, depth_id)
        #    plot_horizontal_slice(perm_field_err[:, :, depth_id], depth_id, str_fig_name, \
        #                          xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)
        #print("i, Train = ", i, counter, "{:.2f}".format(r2), \
        #    "{:.2f}".format(mse), "{:.2f}".format(rmse), \
        #    "{:.2f}".format(mae), "{:.2f}".format(evs), \
        #    "{:.2f}".format(me), "{:.2f}".format(mape), \
        #    "{:.2f}".format(medae), "{:.2f}".format(ssim), \
        #    "{:.2f}".format(pe))

#********************************************************************;
# 7. Estimate kernel density of percent error data (Train/Val/Test)  ;
#********************************************************************;
##https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
##https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
##https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
##https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html
#
num_bins                 = 21
bandwidth                = 0.5
percent_error_grid       = np.linspace(0.0, 25, num_bins) #Percent error grid; min = 0, max = 25
#
train_counts, train_bins = np.histogram(train_err_list.flatten(), bins = num_bins) #Histgram of train data
frac_train_counts        = train_counts/(np.sum(train_counts)) #Normalized frequency
#
val_counts, val_bins     = np.histogram(val_err_list.flatten(), bins = num_bins) #Histgram of val data
frac_val_counts          = val_counts/(np.sum(val_counts)) #Normalized frequency
#
test_counts, test_bins   = np.histogram(test_err_list.flatten(), bins = num_bins) #Histgram of test data
frac_test_counts         = test_counts/(np.sum(test_counts)) #Normalized frequency

#*****************************************************;
# 8. Plot lumped percent error data (Train/Val/Test)  ;
#*****************************************************;
xmin         = 0.0
xmax         = 25.0
ymin         = 0.0
ymax         = 0.6
width_val    = 0.95
hist_label   = 'Training simulations'
loc_pos      = 'best'
str_x_label  = 'Percent error (for each cell)'
str_y_label  = 'Probability'
str_fig_name = path_rand_seed_sav + 'KDE_Hist_Train'
plot_hist_kde_trainvaltest(train_bins, frac_train_counts, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, width_val, str_x_label, str_y_label, str_fig_name)
#
xmin         = 0.0
xmax         = 25.0
ymin         = 0.0
ymax         = 0.6
width_val    = 1.05
hist_label   = 'Validation simulations'
loc_pos      = 'best'
str_x_label  = 'Percent error (for each cell)'
str_y_label  = 'Probability'
str_fig_name = path_rand_seed_sav + 'KDE_Hist_Val'
plot_hist_kde_trainvaltest(val_bins, frac_val_counts, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, width_val, str_x_label, str_y_label, str_fig_name)
#
xmin         = 0.0
xmax         = 25.0
ymin         = 0.0
ymax         = 0.6
width_val    = 1.25
hist_label   = 'Testing simulations'
loc_pos      = 'best'
str_x_label  = 'Percent error (for each cell)'
str_y_label  = 'Probability'
str_fig_name = path_rand_seed_sav + 'KDE_Hist_Test'
plot_hist_kde_trainvaltest(test_bins, frac_test_counts, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, width_val, str_x_label, str_y_label, str_fig_name)

#********************************************;
# 9a. Plot performance metrics (Train data)  ;
#********************************************;
color_list       = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'olive', \
                    'darkgreen', 'gold', 'indigo', 'lavender', 'maroon'] #https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py
marker_list      = ['o', 'v', '^', '<', '>', '1', 's', 'p', 'h', 'H', 'P', '*', '+', 'X', 'D']
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'$R^2$-score'
str_fig_name = path_rand_seed_sav + 'R2_Train'
plot_metrics_realz(np.arange(1,num_train+1), r2_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #R2-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 0.5
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'MSE'
str_fig_name = path_rand_seed_sav + 'MSE_Train'
plot_metrics_realz(np.arange(1,num_train+1), mse_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MSE-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 0.7
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'RMSE'
str_fig_name = path_rand_seed_sav + 'RMSE_Train'
plot_metrics_realz(np.arange(1,num_train+1), rmse_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #RMSE-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 0.4
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Mean absolute error'
str_fig_name = path_rand_seed_sav + 'MAE_Train'
plot_metrics_realz(np.arange(1,num_train+1), mae_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAE-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Explained variance score'
str_fig_name = path_rand_seed_sav + 'EVS_Train'
plot_metrics_realz(np.arange(1,num_train+1), evs_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #EVS-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 4.0
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Maximum error'
str_fig_name = path_rand_seed_sav + 'ME_Train'
plot_metrics_realz(np.arange(1,num_train+1), me_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #ME-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 0.03
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Mean absolute percent error'
str_fig_name = path_rand_seed_sav + 'MAPE_Train'
plot_metrics_realz(np.arange(1,num_train+1), mape_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAPE-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 0.4
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Median absolute error'
str_fig_name = path_rand_seed_sav + 'MEDAE_Train'
plot_metrics_realz(np.arange(1,num_train+1), medae_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MEDAE-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'SSIM'
str_fig_name = path_rand_seed_sav + 'SSIM_Train'
plot_metrics_realz(np.arange(1,num_train+1), ssim_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #SSIM-score plots for train data
#
xmin         = 1.0
xmax         = num_train
ymin         = 4.0
ymax         = 21
loc_pos      = 'best'
str_x_label  = 'Train realizations'
str_y_label  = r'Maximum percent error'
str_fig_name = path_rand_seed_sav + 'PE_Train'
plot_metrics_realz(np.arange(1,num_train+1), pe_train_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #PE-score plots for train data

#******************************************;
# 9b. Plot performance metrics (Val data)  ;
#******************************************;
color_list       = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'olive', \
                    'darkgreen', 'gold', 'indigo', 'lavender', 'maroon'] #https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py
marker_list      = ['o', 'v', '^', '<', '>', '1', 's', 'p', 'h', 'H', 'P', '*', '+', 'X', 'D']
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'$R^2$-score'
str_fig_name = path_rand_seed_sav + 'R2_Val'
plot_metrics_realz(np.arange(1,num_val+1), r2_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #R2-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 0.5
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'MSE'
str_fig_name = path_rand_seed_sav + 'MSE_Val'
plot_metrics_realz(np.arange(1,num_val+1), mse_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MSE-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 0.7
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'RMSE'
str_fig_name = path_rand_seed_sav + 'RMSE_Val'
plot_metrics_realz(np.arange(1,num_val+1), rmse_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #RMSE-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 0.6
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Mean absolute error'
str_fig_name = path_rand_seed_sav + 'MAE_Val'
plot_metrics_realz(np.arange(1,num_val+1), mae_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAE-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Explained variance score'
str_fig_name = path_rand_seed_sav + 'EVS_Val'
plot_metrics_realz(np.arange(1,num_val+1), evs_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #EVS-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 1.8
ymax         = 4.0
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Maximum error'
str_fig_name = path_rand_seed_sav + 'ME_Val'
plot_metrics_realz(np.arange(1,num_val+1), me_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #ME-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 0.03
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Mean absolute percent error'
str_fig_name = path_rand_seed_sav + 'MAPE_Val'
plot_metrics_realz(np.arange(1,num_val+1), mape_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAPE-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 0.4
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Median absolute error'
str_fig_name = path_rand_seed_sav + 'MEDAE_Val'
plot_metrics_realz(np.arange(1,num_val+1), medae_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MEDAE-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'SSIM'
str_fig_name = path_rand_seed_sav + 'SSIM_Val'
plot_metrics_realz(np.arange(1,num_val+1), ssim_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #SSIM-score plots for val data
#
xmin         = 1.0
xmax         = num_val
ymin         = 9.0
ymax         = 26
loc_pos      = 'best'
str_x_label  = 'Validation realizations'
str_y_label  = r'Maximum percent error'
str_fig_name = path_rand_seed_sav + 'PE_Val'
plot_metrics_realz(np.arange(1,num_val+1), pe_val_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #PE-score plots for val data

#*******************************************;
# 9c. Plot performance metrics (Test data)  ;
#*******************************************;
color_list       = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'olive', \
                    'darkgreen', 'gold', 'indigo', 'lavender', 'maroon'] #https://matplotlib.org/stable/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py
marker_list      = ['o', 'v', '^', '<', '>', '1', 's', 'p', 'h', 'H', 'P', '*', '+', 'X', 'D']
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'$R^2$-score'
str_fig_name = path_rand_seed_sav + 'R2_Test'
plot_metrics_realz(np.arange(1,num_test+1), r2_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #R2-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 1.2
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'MSE'
str_fig_name = path_rand_seed_sav + 'MSE_Test'
plot_metrics_realz(np.arange(1,num_test+1), mse_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MSE-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 1.2
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'RMSE'
str_fig_name = path_rand_seed_sav + 'RMSE_Test'
plot_metrics_realz(np.arange(1,num_test+1), rmse_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #RMSE-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 0.8
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Mean absolute error'
str_fig_name = path_rand_seed_sav + 'MAE_Test'
plot_metrics_realz(np.arange(1,num_test+1), mae_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAE-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Explained variance score'
str_fig_name = path_rand_seed_sav + 'EVS_Test'
plot_metrics_realz(np.arange(1,num_test+1), evs_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #EVS-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 1.8
ymax         = 5.0
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Maximum error'
str_fig_name = path_rand_seed_sav + 'ME_Test'
plot_metrics_realz(np.arange(1,num_test+1), me_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #ME-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 0.05
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Mean absolute percent error'
str_fig_name = path_rand_seed_sav + 'MAPE_Test'
plot_metrics_realz(np.arange(1,num_test+1), mape_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MAPE-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 0.75
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Median absolute error'
str_fig_name = path_rand_seed_sav + 'MEDAE_Test'
plot_metrics_realz(np.arange(1,num_test+1), medae_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #MEDAE-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 0.0
ymax         = 1.0
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'SSIM'
str_fig_name = path_rand_seed_sav + 'SSIM_Test'
plot_metrics_realz(np.arange(1,num_test+1), ssim_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #SSIM-score plots for test data
#
xmin         = 1.0
xmax         = num_test
ymin         = 9.0
ymax         = 26
loc_pos      = 'best'
str_x_label  = 'Test realizations'
str_y_label  = r'Maximum percent error'
str_fig_name = path_rand_seed_sav + 'PE_Test'
plot_metrics_realz(np.arange(1,num_test+1), pe_test_list, loc_pos, \
                    xmin, xmax, ymin, ymax, num_random_seeds, \
                    color_list, marker_list, \
                    str_x_label, str_y_label, str_fig_name) #PE-score plots for test data

#*****************************************************;
# 10a. Print metrics over all the realz (Train data)  ;
#*****************************************************;
print('TRAIN R2-score = ', "{:.2f}".format(np.min(r2_train_list)), \
    "{:.2f}".format(np.max(r2_train_list)), \
    "{:.2f}".format(np.mean(r2_train_list)), \
    "{:.2f}".format(np.std(r2_train_list)))
print('TRAIN MSE-score = ', "{:.2f}".format(np.min(mse_train_list)), \
    "{:.2f}".format(np.max(mse_train_list)), \
    "{:.2f}".format(np.mean(mse_train_list)), \
    "{:.2f}".format(np.std(mse_train_list)))
print('TRAIN RMSE-score = ', "{:.2f}".format(np.min(rmse_train_list)), \
    "{:.2f}".format(np.max(rmse_train_list)), \
    "{:.2f}".format(np.mean(rmse_train_list)), \
    "{:.2f}".format(np.std(rmse_train_list)))
print('TRAIN MAE-score = ', "{:.2f}".format(np.min(mae_train_list)), \
    "{:.2f}".format(np.max(mae_train_list)), \
    "{:.2f}".format(np.mean(mae_train_list)), \
    "{:.2f}".format(np.std(mae_train_list)))
print('TRAIN EVS-score = ', "{:.2f}".format(np.min(evs_train_list)), \
    "{:.2f}".format(np.max(evs_train_list)), \
    "{:.2f}".format(np.mean(evs_train_list)), \
    "{:.2f}".format(np.std(evs_train_list)))
print('TRAIN ME-score = ', "{:.2f}".format(np.min(me_train_list)), \
    "{:.2f}".format(np.max(me_train_list)), \
    "{:.2f}".format(np.mean(me_train_list)), \
    "{:.2f}".format(np.std(me_train_list)))
print('TRAIN MAPE-score = ', "{:.2f}".format(np.min(mape_train_list)), \
    "{:.2f}".format(np.max(mape_train_list)), \
    "{:.2f}".format(np.mean(mape_train_list)), \
    "{:.2f}".format(np.std(mape_train_list)))
print('TRAIN MEDAE-score = ', "{:.2f}".format(np.min(medae_train_list)), \
    "{:.2f}".format(np.max(medae_train_list)), \
    "{:.2f}".format(np.mean(medae_train_list)), \
    "{:.2f}".format(np.std(medae_train_list)))
print('TRAIN SSIM-score = ', "{:.2f}".format(np.min(ssim_train_list)), \
    "{:.2f}".format(np.max(ssim_train_list)), \
    "{:.2f}".format(np.mean(ssim_train_list)), \
    "{:.2f}".format(np.std(ssim_train_list)))
print('TRAIN PE-score = ', "{:.2f}".format(np.min(pe_train_list)), \
    "{:.2f}".format(np.max(pe_train_list)), \
    "{:.2f}".format(np.mean(pe_train_list)), \
    "{:.2f}".format(np.std(pe_train_list)))

#***************************************************;
# 10b. Print metrics over all the realz (Val data)  ;
#***************************************************;
print('VAL R2-score = ', "{:.2f}".format(np.min(r2_val_list)), \
    "{:.2f}".format(np.max(r2_val_list)), \
    "{:.2f}".format(np.mean(r2_val_list)), \
    "{:.2f}".format(np.std(r2_val_list)))
print('VAL MSE-score = ', "{:.2f}".format(np.min(mse_val_list)), \
    "{:.2f}".format(np.max(mse_val_list)), \
    "{:.2f}".format(np.mean(mse_val_list)), \
    "{:.2f}".format(np.std(mse_val_list)))
print('VAL RMSE-score = ', "{:.2f}".format(np.min(rmse_val_list)), \
    "{:.2f}".format(np.max(rmse_val_list)), \
    "{:.2f}".format(np.mean(rmse_val_list)), \
    "{:.2f}".format(np.std(rmse_val_list)))
print('VAL MAE-score = ', "{:.2f}".format(np.min(mae_val_list)), \
    "{:.2f}".format(np.max(mae_val_list)), \
    "{:.2f}".format(np.mean(mae_val_list)), \
    "{:.2f}".format(np.std(mae_val_list)))
print('VAL EVS-score = ', "{:.2f}".format(np.min(evs_val_list)), \
    "{:.2f}".format(np.max(evs_val_list)), \
    "{:.2f}".format(np.mean(evs_val_list)), \
    "{:.2f}".format(np.std(evs_val_list)))
print('VAL ME-score = ', "{:.2f}".format(np.min(me_val_list)), \
    "{:.2f}".format(np.max(me_val_list)), \
    "{:.2f}".format(np.mean(me_val_list)), \
    "{:.2f}".format(np.std(me_val_list)))
print('VAL MAPE-score = ', "{:.2f}".format(np.min(mape_val_list)), \
    "{:.2f}".format(np.max(mape_val_list)), \
    "{:.2f}".format(np.mean(mape_val_list)), \
    "{:.2f}".format(np.std(mape_val_list)))
print('VAL MEDAE-score = ', "{:.2f}".format(np.min(medae_val_list)), \
    "{:.2f}".format(np.max(medae_val_list)), \
    "{:.2f}".format(np.mean(medae_val_list)), \
    "{:.2f}".format(np.std(medae_val_list)))
print('VAL SSIM-score = ', "{:.2f}".format(np.min(ssim_val_list)), \
    "{:.2f}".format(np.max(ssim_val_list)), \
    "{:.2f}".format(np.mean(ssim_val_list)), \
    "{:.2f}".format(np.std(ssim_val_list)))
print('VAL PE-score = ', "{:.2f}".format(np.min(pe_val_list)), \
    "{:.2f}".format(np.max(pe_val_list)), \
    "{:.2f}".format(np.mean(pe_val_list)), \
    "{:.2f}".format(np.std(pe_val_list)))

#***************************************************;
# 10c. Print metrics over all the realz (Val data)  ;
#***************************************************;
print('TEST R2-score = ', "{:.2f}".format(np.min(r2_test_list)), \
    "{:.2f}".format(np.max(r2_test_list)), \
    "{:.2f}".format(np.mean(r2_test_list)), \
    "{:.2f}".format(np.std(r2_test_list)))
print('TEST MSE-score = ', "{:.2f}".format(np.min(mse_test_list)), \
    "{:.2f}".format(np.max(mse_test_list)), \
    "{:.2f}".format(np.mean(mse_test_list)), \
    "{:.2f}".format(np.std(mse_test_list)))
print('TEST RMSE-score = ', "{:.2f}".format(np.min(rmse_test_list)), \
    "{:.2f}".format(np.max(rmse_test_list)), \
    "{:.2f}".format(np.mean(rmse_test_list)), \
    "{:.2f}".format(np.std(rmse_test_list)))
print('TEST MAE-score = ', "{:.2f}".format(np.min(mae_test_list)), \
    "{:.2f}".format(np.max(mae_test_list)), \
    "{:.2f}".format(np.mean(mae_test_list)), \
    "{:.2f}".format(np.std(mae_test_list)))
print('TEST EVS-score = ', "{:.2f}".format(np.min(evs_test_list)), \
    "{:.2f}".format(np.max(evs_test_list)), \
    "{:.2f}".format(np.mean(evs_test_list)), \
    "{:.2f}".format(np.std(evs_test_list)))
print('TEST ME-score = ', "{:.2f}".format(np.min(me_test_list)), \
    "{:.2f}".format(np.max(me_test_list)), \
    "{:.2f}".format(np.mean(me_test_list)), \
    "{:.2f}".format(np.std(me_test_list)))
print('TEST MAPE-score = ', "{:.2f}".format(np.min(mape_test_list)), \
    "{:.2f}".format(np.max(mape_test_list)), \
    "{:.2f}".format(np.mean(mape_test_list)), \
    "{:.2f}".format(np.std(mape_test_list)))
print('TEST MEDAE-score = ', "{:.2f}".format(np.min(medae_test_list)), \
    "{:.2f}".format(np.max(medae_test_list)), \
    "{:.2f}".format(np.mean(medae_test_list)), \
    "{:.2f}".format(np.std(medae_test_list)))
print('TEST SSIM-score = ', "{:.2f}".format(np.min(ssim_test_list)), \
    "{:.2f}".format(np.max(ssim_test_list)), \
    "{:.2f}".format(np.mean(ssim_test_list)), \
    "{:.2f}".format(np.std(ssim_test_list)))
print('TEST PE-score = ', "{:.2f}".format(np.min(pe_test_list)), \
    "{:.2f}".format(np.max(pe_test_list)), \
    "{:.2f}".format(np.mean(pe_test_list)), \
    "{:.2f}".format(np.std(pe_test_list)))

"""
#********************************************;
# 11. KDE vs percent error (DL predictions)  ;
#********************************************;
num_bins                = 101
num_grid_points         = np.linspace(0.0, 25, num_bins) #percent error
#
kde_pe_train_fit         = sm.nonparametric.KDEUnivariate(train_err_list.flatten()[0:-1:5])
kde_pe_train_fit.fit() # Estimate the densities for pe
kde_pe_train_grid        = kde_pe_train_fit.evaluate(num_grid_points) #KDE on train grid
#
kde_pe_val_fit           = sm.nonparametric.KDEUnivariate(val_err_list.flatten())
kde_pe_val_fit.fit() # Estimate the densities for pe
kde_pe_val_grid          = kde_pe_val_fit.evaluate(num_grid_points) #KDE on val grid
#
kde_pe_test_fit          = sm.nonparametric.KDEUnivariate(test_err_list.flatten())
kde_pe_test_fit.fit() # Estimate the densities for pe
kde_pe_test_grid         = kde_pe_test_fit.evaluate(num_grid_points) #KDE on test grid
#
xmin         = 0.0
xmax         = 25.0
ymin         = 0.0
ymax         = 3.0
loc_pos      = 'upper right'
str_x_label  = 'Percent error (for each cell)'
str_y_label  = 'KDE'
str_fig_name = path_rand_seed_sav + 'KDE_Percent_Error'
plot_kde_trainvaltest(num_grid_points, kde_pe_train_grid, kde_pe_val_grid, kde_pe_test_grid, \
    xmin, xmax, ymin, ymax, loc_pos, str_x_label, str_y_label, str_fig_name)
"""

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)