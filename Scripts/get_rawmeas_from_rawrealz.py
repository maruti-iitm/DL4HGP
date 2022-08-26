# Get raw measurement *.npy from raw realz *.npy for all 302 realz and 40466 measurements (new data generated on June-10th)
#   size of raw realz *.npy file is: (29,40467)
#   size of raw meas *.npy file is: (28,302)
# 302_June10_raw_npy/Potential_Data_R*.npy --> 302_June10_rawmeas_npy/Sim_new_*.npy

import numpy as np
import time

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#**********************************;
# 1. Set new 302 realz data paths  ; 
#**********************************;
path_new_data     = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/New_Data/"
path_raw_npy      = path_new_data + "302_June10_raw_npy/" #302 raw data realz in .npy format
path_rawmeas_npy  = path_new_data + "302_June10_rawmeas_npy/" #40466 raw measurements in .npy format

#********************************************************************;
# 2. Initialize and convert raw realz *.npy to raw meas *.npy files  ; 
#********************************************************************;
num_time_stamps = 28 #Total no. of time-stamps
num_pots        = 40466 #Total no. of potential measurements
num_realz       = 302 #Total no. of new realz
#
pots_list       = list(range(0,num_pots)) #40466 length
realz_list      = list(range(1,num_realz+1)) #302 realizations
#
for pots in pots_list: #Iterate over 40466 measurements
    meas_data = np.zeros((num_time_stamps,num_realz)) #(28,302)
    #
    for realz in realz_list: #Iterate over 302 realz
        v_new                = np.load(path_raw_npy + 'Potential_Data_R' + \
										str(realz) + '.npy')[0:num_time_stamps,1:] #(28,40466)
        meas_data[:,realz-1] = v_new[:,pots] #Get pots-th measurement data for each realz
        print("pots, realz = ", pots, realz)

    np.save(path_rawmeas_npy + "Sim_new_" + str(pots) + ".npy", meas_data) #Save each pots file (28,302)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)