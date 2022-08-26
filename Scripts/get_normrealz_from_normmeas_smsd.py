# ACROSS ALL 28 TIME-STAMPS, WE GET SINGLE MEAN AND SINGLE STD
#   302 different means and 302 different standard deviations for each measurement
#   (302,40466) mean values
#   (302,40466) standard deviations values
# Get normalized realz *.npy from normalized measurement *.npy (new data generated on June-10th)
#   size of norm meas *.npy file is: (28,302)
#   size of norm realz *.npy file is: (28,40466)
# 302_June10_normmeas_smsd_npy/Sim_new_smsdnorm_*.npy ---> 302_June10_norm_smsd_npy/Normsmsd_Potential_Data_R*.npy

import numpy as np
import time

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#**********************************;
# 1. Set new 302 realz data paths  ; 
#**********************************;
path_new_data          = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/New_Data/"
path_raw_npy           = path_new_data + "302_June10_raw_npy/" #302 raw data realz in .npy format
path_rawmeas_npy       = path_new_data + "302_June10_rawmeas_npy/" #40466 raw measurements in .npy format
path_normmeas_smsd_npy = path_new_data + "302_June10_normmeas_smsd_npy/" #40466 norm measurements by smsd in .npy format
path_norm_smsd_npy     = path_new_data + "302_June10_norm_smsd_npy/" #302 norm data realz by smsd in .npy format

#***************************************************************************;
# 2. Initialize and convert norm meas smsd *.npy to norm realz *.npy files  ; 
#***************************************************************************;
num_time_stamps = 28 #Total no. of time-stamps
num_pots        = 40466 #Total no. of potential measurements
num_realz       = 302 #Total no. of new realz
#
pots_list       = list(range(0,num_pots)) #40466 length
realz_list      = list(range(1,num_realz+1)) #302 realizations
#
for realz in realz_list: #Iterate over 302 realz
    new_realz_data_norm = np.zeros((num_time_stamps,num_pots)) #Normalized new realz norm data (28,40466) by smsd
    #
    for pots in pots_list: #Iterate over 40466 measurements
        new_meas_data_norm          = np.load(path_normmeas_smsd_npy + "Sim_new_smsdnorm_" + str(pots) + ".npy") #Each norm pots file (28,302) by smsd
        new_realz_data_norm[:,pots] = new_meas_data_norm[:,realz-1] #Get realz-th norm measurement data for each pots by smsd
        print("pots, realz = ", pots, realz)

    np.save(path_norm_smsd_npy + "Normsmsd_Potential_Data_R" + str(realz) + ".npy", \
			new_realz_data_norm) #Save each normalized new realz file (28,40466); 302 of them

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)