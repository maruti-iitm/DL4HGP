# Get indices for all 302 realizations
#   TRAINING:   282 realz
#   VALIDATION: 10  realz (same as old 200 realz indices) [106, 112, 118, 124, 179, 187, 6, 69, 85, 87]
#   TESTING:    10  realz (same as old 200 realz indices) [10, 152, 168, 174, 195, 5, 55, 75, 79, 81]
# AUTHOR: Maruti Kumar Mudunuru

import os
import time
import copy
import numpy as np

#******************************;
# 0. Set data paths (INDICES)  ; 
#******************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders

#************************************;
#  1. Get val and test indices list  ;
#************************************;
all_index_list     = np.arange(1,303) #All realz indices
val_index_list     = np.genfromtxt(path + "Raw_Data/0_PreProcessed_Data/Train_Val_Test_Indices/" + \
                                       "Val_Realz.txt", dtype = int, skip_header = 1) #Val indices
test_index_list    = np.genfromtxt(path + "Raw_Data/0_PreProcessed_Data/Train_Val_Test_Indices/" + \
                                       "Test_Realz.txt", dtype = int, skip_header = 1) #Test indices
valtest_index_list = np.concatenate((val_index_list, test_index_list), axis = 0) #Val and test indices
train_index_list   = np.array(list(set(all_index_list) - set(valtest_index_list)), dtype = int) #Train indices

#***********************************************************;
#  2. Save train, val, and test indices list for 302 realz  ;
#***********************************************************;
np.savetxt(path_proccessed_data + "Train_Val_Test_Indices/Train_Realz.txt", \
           train_index_list, fmt = '%d', header = 'Train_realizations_IDs') #Save training indices (282,)
np.savetxt(path_proccessed_data + "Train_Val_Test_Indices/Val_Realz.txt", \
           val_index_list, fmt = '%d', header = 'Val_realizations_IDs') #Save validation indices (10,)
np.savetxt(path_proccessed_data + "Train_Val_Test_Indices/Test_Realz.txt", \
           test_index_list, fmt = '%d', header = 'Test_realizations_IDs') #Save test indices (10,)