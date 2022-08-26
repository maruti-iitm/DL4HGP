# MIXUP on 302 REALZ DATA (see get_mixup_PCA_160.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Pre-computed mixup on PCA components -- Generate weak supervised labels
# Mixup lambda is from Beta distribution (https://arxiv.org/pdf/1710.09412.pdf)
# alpha > 0, beta > 0; we choose alpha = 0.5 and beta = 0.5
# Author: Maruti Kumar Mudunuru
# Update: March-18-2021 (n_comps = 160)

import time
import numpy as np
import pandas as pd
from itertools import combinations

#Start time
start_time = time.time()
#==================================================;
#  Function-1: Pre-computed mixup for two samples  ;
#==================================================;
def mixup_data(x1, x2, y1, y2, lam):

    #--------------------------------;
    # Generate mixup of two samples  ;
    #--------------------------------;
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little')) #Needed for multiprocessing -- Don't commment it
    #lam   = np.random.beta(alpha, beta) #Sample from beta distribution #np.random.dirichlet((1, 1, 1), 20) for multi-dimensional mixup
    x_lam = lam * x1 + (1.0 - lam) * x2 #Mixup inputs vectors
    y_lam = lam * y1 + (1.0 - lam) * y2 #Mixup output vectors

    return x_lam, y_lam, lam

#==========================================================;
#  Function-2: Compute all unique combination from a list  ;
#              (If a len(list) = n, it will be n*(n-1)/2   ;
#               unique combinations)                       ;
#==========================================================;
def get_all_unique_pairs(num_realz, k):
   
    #----------------------------------------------------------------;
    # Generate unique pairs of realizations                          ;
    # (nCk = n! / k! / (n-k)! when 0 <= k <= n or zero when k > n.)  ; 
    #----------------------------------------------------------------;
    realz_list = list(range(0,num_realz)) #List of realization numbers
    comb_list  = np.array([comb for comb in \
                          combinations(realz_list,k)]) #List of all possible unique pairs
    
    return comb_list

#***************************************************************************;
#  Set paths, load preprocessed encoded PCA data and dump .csv mixup files  ;
#***************************************************************************;
#
if __name__ == '__main__':
    
    #-------------------------------------------------------------------;
    #  1. Get pre-processed data (training perm PCA and pot PCA comps)  ;
    #-------------------------------------------------------------------;
    #path = os.getcwd() #Get current directory path
    path                  = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
    path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
    #
    train_pots_fl         = path_proccessed_data + "2_PCA_comps/train_pca_comp_potential.csv"
    train_perm_fl         = path_proccessed_data + "2_PCA_comps/train_pca_comp_permeability_unscaled.csv"

    #-----------------------------------;
    #  2. Train PCA perm and pots data  ;
    #-----------------------------------;
    df_train_pots        = pd.read_csv(train_pots_fl, index_col = 0) #PCA train pot comps (282, 270)
    df_train_perm        = pd.read_csv(train_perm_fl, index_col = 0) #PCA train perm comps (282, 246)
    #
    train_pots           = df_train_pots.values #Training PCA pots data (282, 270)
    train_perm           = df_train_perm.values #Training PCA perm data (282, 246)
    #
    npotential_comps     = df_train_pots.shape[1] #No. of PCA pot comps #270
    nperm_comps          = df_train_perm.shape[1] #No. of PCA perm comps #246
    num_realz            = df_train_pots.shape[0] #282 realizations
    #
    pot_comps_cols_list  = df_train_pots.columns.to_list() #Pots PCA list -- A total of 270 comps 
    perm_comps_cols_list = df_train_perm.columns.to_list() #Perm PCA list -- A total of 246 comps

    #----------------------------------------------------------------------------;
    #  3. Generate all possible unique pairs of indices for (x1,y1) and (x2,y2)  ;
    #     (get lambda values from Beta or Dirichlet distribution)                ;
    #----------------------------------------------------------------------------;
    np.random.seed(1337)
    #
    comb_list = get_all_unique_pairs(num_realz, k = 2) #Unique pairs
    print(comb_list)
    print(comb_list.shape, int(0.5*num_realz*(num_realz-1))) #size (0.5*n*(n-1),2) (39621, 2)
    #
    alpha = 0.5 #0.1 to 0.5
    beta  = 0.5 #0.1 to 0.5
    #
    num_lam  = 10 #10-100 random numbers for each mixup sample
    num_comb = comb_list.shape[0] #39621
    #
    lam_list = np.array([np.random.beta(alpha, beta) for i in range(0,num_lam)]) #10 beta distribution values
    print("Min and Max lam values = ", np.min(lam_list), np.max(lam_list))
    #
    np.savetxt(path_proccessed_data + "Comb_Mixup_Indices/Comb_List_Indices.txt", \
               comb_list, fmt = '%d', header = 'All possible combination (2 pairs)') #Save combination indices (39621, 2)
    np.savetxt(path_proccessed_data + "Comb_Mixup_Indices/Mixup_Lambda_Values.txt", \
               lam_list, header = 'Lambda values (Beta distribution)') #Save Mixup lambda values (10,1)
    #
    comb_list = np.genfromtxt(path_proccessed_data + "Comb_Mixup_Indices/Comb_List_Indices.txt", \
                              dtype = int, skip_header = 1) #combination indices (39621, 2)
    lam_list  = np.genfromtxt(path_proccessed_data + "Comb_Mixup_Indices/Mixup_Lambda_Values.txt", \
                              skip_header = 1) #Mixup lambda values (10,1)

    #-------------------------------------------------------------------------;
    #  4. Generate mixup samples for potential and permeability PCA comps     ;
    #     PCA pot comps = (39621*10, 270); PCA perm comps = (39621*10, 246)   ;
    #-------------------------------------------------------------------------;
    counter          = 0
    train_pots_mixup = np.zeros((num_comb*num_lam,npotential_comps)) #(39621*10, 270)
    train_perm_mixup = np.zeros((num_comb*num_lam,nperm_comps)) #(39621*10, 246)
    #
    for i in range(0,num_comb): #Iterate over all combinations
        index1, index2 = comb_list[i] #Combination indices
        #
        for j in range(0,num_lam): #Iterate over all lambda's
            lam = lam_list[j] #lambda value
            #
            x1  = train_pots[index1] #pots-1
            x2  = train_pots[index2] #pots-2
            #
            y1  = train_perm[index1] #perm-1
            y2  = train_perm[index2] #perm-2
            #
            x_lam, y_lam, lam = mixup_data(x1, x2, y1, y2, lam) #Mixup encoded PCA pots and perms
            #
            train_pots_mixup[counter,:] = x_lam #Pots
            train_perm_mixup[counter,:] = y_lam #Pots
            #
            print("counter, i, j, index1, index2, lam = ", counter, i, j, index1, index2, lam)
            counter = counter + 1

    #-------------------------------------------------------------------------;
    #  5. Save mixup pots and perm PCA comps                                  ;
    #     PCA pot comps = (39621*10, 270); PCA perm comps = (39621*10, 246)   ;
    #-------------------------------------------------------------------------;
    #train_pots_mixup_df = pd.DataFrame(train_pots_mixup, index=list(range(0,num_comb*num_lam)), \
    #                                   columns = pot_comps_cols_list) #[396210 rows x 270 columns]
    #
    #train_perm_mixup_df = pd.DataFrame(train_perm_mixup, index=list(range(0,num_comb*num_lam)), \
    #                                   columns = perm_comps_cols_list) #[396210 rows x 246 columns]
    #
    #train_pots_mixup_df.to_csv(path_proccessed_data + \
    #                           "4_PCA_Mixup/train_pca_comp_potential_mixup.csv") #Save train PCA comp mixup pots [396210 rows x 270 columns]
    #train_perm_mixup_df.to_csv(path_proccessed_data + \
    #                           "4_PCA_Mixup/train_pca_comp_permeability_mixup.csv") #Save train PCA comp mixup perm [396210 rows x 246 columns]
    np.save(path_proccessed_data + "4_PCA_Mixup/train_pca_comp_potential_mixup.npy", \
            train_pots_mixup) #Save PCA mixup pots data [396210 rows x 270 columns]
    np.save(path_proccessed_data + "4_PCA_Mixup/train_pca_comp_permeability_mixup.npy", \
            train_perm_mixup) #Save PCA mixup perm data [396210 rows x 246 columns]


end_time   = time.time() #End timing
combo_time = end_time - start_time # Calculate total runtime
print("Total time = ", combo_time)