# PROCESS 302 REALZ DATA (see process_data.py for 200 realz)
# Paths for NEW data (*.npy files)
# MASKING CELLS:
#   (A) river cells:   ID = 0
#   (B) Ringold cells: ID = 4
#   (C) Cells from:    ID = -1 (from 107m to 110m; in new *.npy masking file)
# Process data and save the normalized data 
#    PERMEABILITY DATA: We have two different options. We will use OPTION-1.
#                       OPTION-1: ln[k] and no normalization needed. Values are important
#                       OPTION-2: StandardScalar for each permeability realization
#    ELECTRICAL POTENTIAL DATA: Normalize by measurement smsd. 
#                               Each measurement for a given realz has mean = 0 and std = 1 across all time-stamps.                            
# AUTHOR: Maruti Kumar Mudunuru

import os
import re
import copy
import pickle
import time
import numpy as np
import pandas as pd
#
from sklearn.preprocessing import StandardScaler

#Start time
start_time = time.time()
#***************************************************;
# 0. Set OLD data paths (raw and normalized *.npy)  ; 
#***************************************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_mask_id_npy     = path + "302_Realz_Models/Material_IDs_Plume_2mx2mxhalfm.npy" #Original mask *.npy file (IDs = 0,1,4)
path_perm_npy        = path + "Raw_Data/Prior_Perm_Plume_HGP2013_Rel302_npy/" #302 perm *.npy files
path_norm_smsd_npy   = path + "New_Data/302_June10_norm_smsd_npy/" #302 norm data realz by smsd in .npy format
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
path_ss_pca_sav      = path + "302_Realz_Models/1_SS_PCA_models/" #Saved StandardScalar and PCA models

#***************************************************************************************;
#  1. Masking cells: We have Cell IDs = 0 (river cells),                                ; 
#                 1 (Hanford cells), 4 (ringold cells), and                             ;                 
#                 -1 (z = 107m to 109.5m, cells above the watertable fluctuating zone)  ;
#***************************************************************************************;
mask_id_list       = np.load(path_mask_id_npy) #(1600000,)
num_cells          = mask_id_list.shape[0] #(1600000,)
#
nan_cells          = mask_id_list == 0 #(1600000,) -- River cells
tiny_cells         = mask_id_list == 4 #(1600000,) -- Ringold cells
hf_cells           = mask_id_list == 1 #(1600000,) -- Hanford cells
#
nan_cells_id       = np.argwhere(mask_id_list == 0)[:,0] #(63856,) River cells IDs
tiny_cells_id      = np.argwhere(mask_id_list == 4)[:,0] #(742150,) Ringold cells IDs
hf_cells_id        = np.argwhere(mask_id_list == 1)[:,0] #(793994,) Hanford cells IDs
num_mask_cells     = nan_cells_id.shape[0] + tiny_cells_id.shape[0] #806006 masked cells
num_non_mask_cells = num_cells - num_mask_cells #793994, which are all the Hanford cells
#
print('nan_cells_id = ', nan_cells_id, nan_cells_id.shape)
print('tiny_cells_id = ', tiny_cells_id, tiny_cells_id.shape)
print('hf_cells_id = ', hf_cells_id, hf_cells_id.shape)
print('num_mask_cells, non-mask-cells = ', num_mask_cells, num_non_mask_cells)
#
temp                              = copy.deepcopy(mask_id_list) #mask_id_list is not changed
cell_id_array                     = temp.reshape((200, 200, 40)) #(200, 200, 40)
z_index_start                     = 34 #depth = 107m
z_index_end                       = 39 #depth = 109.5m (is the last depth in the file (index 39))
cell_id_array[:,:,z_index_start:] = -1 #A total of 200 x 200 x 6 = 240000 cells
#
new_mask_id_list       = cell_id_array.flatten() #We have Cell IDs = 0, 1 (Hanford), 4, and -1 (107m to 109.5m)
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
#
np.save(path + "302_Realz_Models/New_Material_IDs_Plume_2mx2mxhalfm.npy", new_mask_id_list) #Save new masked cells in *.npy file

#******************************************************************;
#  2. Load permeability data and mask it.                          ;
#     Save pre-processed and post-processed perm data (302 realz)  ; 
#******************************************************************;
num_realz     = 302 #All perm realz
hf_perm_realz = np.zeros((num_realz,num_new_non_mask_cells)) #(302,585453) (num_samples,num_perm_features)
#
for realz in range(1,num_realz+1):
    perm                     = np.load(path_perm_npy + "Permeability" + \
                                       str(realz)+ ".npy") #Unscaled perm data #(200, 200, 40)
    perm_flat                = perm.flatten() #(1600000,) Unscaled values
    hf_perm_flat             = perm_flat[new_hf_cells_id] #(585453,) new hanford perm cells (Unscaled values)
    hf_perm_realz[realz-1,:] = hf_perm_flat #(302,585453)
    print('perm realz = ', realz)

np.save(path_proccessed_data + "0_Raw/Permeability_" + str(num_realz) + \
        "_" + str(num_new_non_mask_cells) + ".npy", hf_perm_realz) #Save unscaled perm data (302,585453) 

print("Min and Max perm realz = ", np.min(hf_perm_realz), np.max(hf_perm_realz)) #Min = 8.63e-12, Max = 1.48e-06
print("Mean and std of perm realz = ", np.min(hf_perm_realz), np.max(hf_perm_realz)) #Mean = 5.33e-09, STD = 1.21e-08

#*********************************************;
#  3. Natural log transform (i.e., ln[perm])  ;
#*********************************************;
ln_hf_perm_realz = np.log(np.load(path_proccessed_data + "0_Raw/Permeability_" + \
                                    str(num_realz) + "_" + str(num_new_non_mask_cells) + \
                                    ".npy"))#Load unscaled perm data (302,585453) and natural log transform it
print("Min and Max ln[K] realz = ", np.min(ln_hf_perm_realz), np.max(ln_hf_perm_realz)) #Min = -25.475, Max = -13.420
print("Mean and std of ln[K] realz = ", np.min(ln_hf_perm_realz), np.max(ln_hf_perm_realz)) #Mean = -19.445, STD = 0.781
#
np.save(path_proccessed_data + "0_Raw/ln_Permeability_" + str(num_realz) + \
          "_" + str(num_new_non_mask_cells) + ".npy", ln_hf_perm_realz) #Save unscaled ln[k] data (302,585453)

#**************************************************************************;
#  4. Load, reshape, and normalize potential data and save it (302 realz)  ;
#     (Normalize by measurement smsd for all 40466 measurements)           ;
#     (Each measurement has mean = 0 and std = 1 across all time-stamps)   ;
#**************************************************************************;
num_time_stamps = 28 #Total no. of time-stamps
num_pots        = 40466 #Total no. of potential measurements
num_realz       = 302 #Total number of realizations (new data)
#
norm_pot_realz  = np.zeros((num_realz,num_time_stamps*num_pots)) #(302,1133048) (num_samples,num_pot_features)
#
for realz in range(1,num_realz+1):
    norm_potential_data       = np.load(path_norm_smsd_npy + "Normsmsd_Potential_Data_R" + str(realz) + ".npy") #(28,40466)
    norm_potential_flat       = norm_potential_data.flatten() #(1133048,) Potential values (norm by measure)
    norm_pot_realz[realz-1,:] = norm_potential_flat #(302,1133048) Potential values (norm by measure)
    print("New normalized pots realz = ", realz)

np.save(path_proccessed_data + "1_Normalized/NormByMeas_Potential_" + str(num_realz) + \
        "_" + str(num_time_stamps*num_pots) + ".npy", norm_pot_realz) #Save norm by measure potential data (302,1133048)

end_time   = time.time() #End timing
combo_time = end_time - start_time # Calculate total runtime
print("Total time = ", combo_time)