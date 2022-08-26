# ACROSS ALL 28 TIME-STAMPS, WE GET SINGLE MEAN AND SINGLE STD
#   302 different means and 302 different standard deviations for each measurement
#   (302,40466) mean values
#   (302,40466) standard deviations values
# Get normalized by measurement *.npy from raw measurement *.npy for all 40466 measurements (new data generated on June-10th)
#   size of raw meas *.npy file is: (28,302)
#   size of norm meas *.npy file is: (28,302)
# 302_June10_rawmeas_npy/Sim_new_*.npy ---> 302_June10_normmeas_smsd_npy/Sim_new_smsdnorm_*.npy
# smsd --> single-mean-single-standarddeviation

import numpy as np
import time

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#====================================================================;
#  Function-1: Stats on time-series measurement data for each realz  ;
#====================================================================;
def get_ERT_smsd_stats(y_mat_data, num_realz):

    #------------------------------------------------------------------;
    #  Compute y_mean and y_std for each realz across all time-levels  ;
    #------------------------------------------------------------------; 
    y_mean_list = np.zeros(num_realz) #Mean #(302,)
    y_std_list  = np.zeros(num_realz) #Standard deviation #(302,)
    #
    for i in range(0,num_realz): #Iterate over 302 realz
        y_mean_list[i] = np.mean(y_mat_data[:,i]) #Mean for each realz across all 28 time-stamps
        y_std_list[i]  = np.std(y_mat_data[:,i]) #STD for each realz across all 28 time-stamps

    return y_mean_list, y_std_list #(302,) and (302,)

#**********************************;
# 1. Set new 302 realz data paths  ; 
#**********************************;
path_new_data          = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/New_Data/"
path_raw_npy           = path_new_data + "302_June10_raw_npy/" #302 raw data realz in .npy format
path_rawmeas_npy       = path_new_data + "302_June10_rawmeas_npy/" #40466 raw measurements in .npy format
path_normmeas_smsd_npy = path_new_data + "302_June10_normmeas_smsd_npy/" #40466 norm measurements by smsd in .npy format

#*************************************************************************;
# 2. Initialize and convert raw meas *.npy to norm meas smsd *.npy files  ; 
#*************************************************************************;
num_time_stamps = 28 #Total no. of time-stamps
num_pots        = 40466 #Total no. of potential measurements
num_realz       = 302 #Total no. of new realz
#
pots_list       = list(range(0,num_pots)) #40466 length
realz_list      = list(range(1,num_realz+1)) #302 realizations
#
for pots in pots_list: #Iterate over 40466 measurements
    new_meas_data             = np.load(path_rawmeas_npy + "Sim_new_" + str(pots) + ".npy") #Each raw pots file (28,302)
    yn_mean_list, yn_std_list = get_ERT_smsd_stats(new_meas_data, num_realz) #Mean and STD (302,) and (302,)
    new_meas_data_norm        = np.zeros((num_time_stamps,num_realz)) #Normalized pots data (28,302)
    #
    for realz in range(0,num_realz):
        yn_mean                     = yn_mean_list[realz] #Mean for the specific measurement (among 40466) and specific realz (among 302)
        yn_std                      = yn_std_list[realz] #STD for the specific measurement (among 40466) and specific realz (among 302)
        new_meas_data_norm[:,realz] = (new_meas_data[:,realz] - yn_mean)/yn_std #(28,302)

    np.save(path_normmeas_smsd_npy + "Sim_new_smsdnorm_" + str(pots) + ".npy", \
			new_meas_data_norm) #Save each new_meas_data_norm pots file (28,302)

    print('Norm pots = ', pots)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)