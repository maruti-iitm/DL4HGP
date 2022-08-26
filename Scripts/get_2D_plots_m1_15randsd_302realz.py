# 2D contour plots for 302 REALZ DATA (see get_2D_plots_m1_160_15randsd.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Generate 2D slices of the permeability field (Ground truth vs. predictions)
# Best set of 15 models
# Inputs  -- encoded electrical potential difference measurements (Var = 0.99; n_comps = 270)
# Outputs -- encoded permeability field (Var = 0.99; n_comps = 246)
# Analysis is performed for No-Mixup --> m1_15randsd_302realz_mp.py
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

#******************************************************************;
# 0. Set OLD data paths (ln[K] ground truth and prediction *.npy)  ; 
#******************************************************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
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
    vmin          = -25 #ln[K] min value
    vmax          = -16 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    #
    for counter in list(range(0,num_test)):
        realz_indx   = test_index_list[counter] #Test index
        str_fig_name = path_fl_sav + "Test/" + str(realz_indx) + "_test_pred_"
        perm_flat    = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat[new_tiny_cells_id] = -25 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat[new_z_cells_id]    = -25 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat[new_hf_cells_id]   = test_perm_pred[counter,:] #(585453,) New Hanford cells IDs
        #
        perm_field                   = perm_flat.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        for depth_id in depth_id_list:
            print("Test counter, realz, depth = ", counter, realz_indx, depth_id)
            plot_horizontal_slice(perm_field[:, :, depth_id], depth_id, str_fig_name, \
                                  xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)

    """
    #************************************;
    # 5. Validation plots (predictions)  ;
    #************************************;
    xticklabels   = 20
    yticklabels   = 20
    vmin          = -25 #ln[K] min value
    vmax          = -16 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    #
    for counter in list(range(0,num_val)):
        realz_indx   = val_index_list[counter] #Validation index
        str_fig_name = path_fl_sav + "Val/" + str(realz_indx) + "_val_pred_"
        perm_flat    = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat[new_tiny_cells_id] = -25 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat[new_z_cells_id]    = -25 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat[new_hf_cells_id]   = val_perm_pred[counter,:] #(585453,) New Hanford cells IDs
        #
        perm_field                   = perm_flat.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        for depth_id in depth_id_list:
            #print("Val counter, realz, depth = ", counter, realz_indx, depth_id)
            plot_horizontal_slice(perm_field[:, :, depth_id], depth_id, str_fig_name, \
	                              xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)

    #**********************************;
    # 6. Training plots (predictions)  ;
    #**********************************;
    xticklabels   = 20
    yticklabels   = 20
    vmin          = -25 #ln[K] min value
    vmax          = -16 #ln[K] max value 
    cmap          = "jet"
    depth_id_list = list(range(0,40))#Depth id goes from 0 to 33 (90m to 107m); ids from 34 to 39 are neglected
    counter_list  = [0, 152, 10, 101] #np.argwhere(train_index_list == 1)
    #
    #for counter in list(range(0,num_train)):
    for counter in counter_list:
        realz_indx   = train_index_list[counter] #Training index
        str_fig_name = path_fl_sav + "Train/" + str(realz_indx) + "_train_pred_"
        perm_flat    = np.zeros(num_cells) #(1600000,) initialization to zeros
        #
        perm_flat[new_nan_cells_id]  = np.nan #NAN value (32397,) River cells (Cell ID = 0)
        perm_flat[new_tiny_cells_id] = -25 #(742150,) Ringold cells (Cell ID = 4)
        perm_flat[new_z_cells_id]    = -25 #(240000,) z = 107m to 109.5m (Cell ID = -1)
        #
        perm_flat[new_hf_cells_id]   = train_perm_pred[counter,:] #(585453,) New Hanford cells IDs
        #
        perm_field                   = perm_flat.reshape((200,200,40)) #Rescale perm data  to (200,200,40)
        #
        for depth_id in depth_id_list:
            #print("Train counter, realz, depth = ", counter, realz_indx, depth_id)
            plot_horizontal_slice(perm_field[:, :, depth_id], depth_id, str_fig_name, \
                                  xticklabels, yticklabels, vmin, vmax, realz_indx, cmap)
    """

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)