# PCA on 302 REALZ DATA (see get_PCA_plots.py for 200 realz)
# Paths for NEW data (*.npy files)
#
# Plot PCA explained variance for potential and permeability (un-scaled)
# PERMEABILITY: (Var = 0.99; n_comps = 246)
# POTENTIAL:    (Var = 0.99; n_comps = 270) 
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

#========================================================;
#  Function-1: Plot lump distribution of percent errors  ;
#              (Histogram vs KDE)                        ;
#========================================================;
def plot_hist_PCA_EVR(y_data, num_bins, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, str_x_label, str_y_label, str_fig_name):

    #------------------------------;
    #  histogram of PCA variances  ;
    #------------------------------;
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
    ax.bar(y_data[:,0]+1, y_data[:,1], align = 'center', \
            label = hist_label, edgecolor = 'k', alpha = 0.5)
    tick_spacing = 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    #fig.savefig(str_title + '.pdf')
    fig.savefig(str_fig_name + '.png')

#*************************************************************;
# 0. Set OLD data paths (pre-processed and normalized *.npy)  ; 
#*************************************************************;
path                 = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/"
path_proccessed_data = path + "302_Realz_Models/0_PreProcessed_Data/" #All pre-processed data folders
path_ss_pca_sav      = path + "302_Realz_Models/1_SS_PCA_models/" #Saved StandardScalar and PCA models
path_PCA_pred_npy    = path + "302_Realz_Models/5_PCA_Predictions_Data/" #PCA inverse transform (reconstructed) data
#
df_ex_pot_var        = pd.read_csv(path_proccessed_data + "2_PCA_comps/potential_explained_ratio_variance.csv", \
                                    header = None) #Explained variance ratio of potential PCA data [270 rows x 1 columns]
#
df_ex_perm_us_var    = pd.read_csv(path_proccessed_data + "2_PCA_comps/perm_unscaled_explained_ratio_variance.csv", \
                                    header = None) #Explained variance ratio of unscaled ln[K] PCA data [246 rows x 1 columns]
#
ex_pot_var           = df_ex_pot_var.values #Potential explained variance #(270, 2)
ex_perm_us_var       = df_ex_perm_us_var.values #ln[K] unscaled explained variance (246,2)

#
num_bins     = 270
xmin         = 0
xmax         = 270
ymin         = 0
ymax         = 0.15
hist_label   = 'Potential field'
str_x_label  = 'Principal components' 
str_y_label  = 'Explained variance ratio'
str_fig_name = path_proccessed_data + 'PCA_Potential_EVS'
loc_pos      = 'best'
plot_hist_PCA_EVR(ex_pot_var, num_bins, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, str_x_label, str_y_label, str_fig_name)
#
num_bins     = 246
xmin         = 0
xmax         = 246
ymin         = 0
ymax         = 0.45
hist_label   = r'Permeability field:~$\ln[K]$'
str_x_label  = 'Principal components' 
str_y_label  = 'Explained variance ratio'
str_fig_name = path_proccessed_data + 'PCA_Perm_EVS'
loc_pos      = 'best'
plot_hist_PCA_EVR(ex_perm_us_var, num_bins, xmin, xmax, ymin, ymax, \
    hist_label, loc_pos, str_x_label, str_y_label, str_fig_name)

end_time   = time.time() #End timing
combo_time = end_time - start_time # Calculate total runtime
print("Total time = ", combo_time)