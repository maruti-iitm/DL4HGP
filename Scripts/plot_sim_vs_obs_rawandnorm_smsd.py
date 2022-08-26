# ACROSS ALL 28 TIME-STAMPS, WE GET SINGLE MEAN AND SINGLE STD
#   302 different means and 302 different standard deviations for each measurement
#   (302,40466) mean values
#   (302,40466) standard deviations values
# Plot normalized measurement by smsd *.npy vs normalized obs data (new data generated on June-10th)
# Plot raw measurement *.npy vs raw obs data (new data generated on June-10th)
#   size of norm/raw meas *.npy file is: (28,302), (28,302)
#   size of norm/raw obs *.txt file is: (28,)
# 302_June10_rawmeas_npy/Sim_new_*.npy and 302_June10_normmeas_smsd_npy/Sim_new_smsdnorm_*.npy

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#=============================================================;
#  Function-1: Plot new vs. obs potential data (time-series)  ;
#=============================================================;
def plot_pots_vs_obs(data_list, obs_data_list, t_list, pots, \
	                 str_x_label, str_y_label, str_fig_name, \
                     col_data):

    #-----------------------------------------;
    #  Time-series realizations vs. obs data  ;
    #-----------------------------------------;
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
    plt.title('Measurement-' + str(pots+1), fontsize = 24, fontweight = 'bold')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #
    num_rows, num_cols = data_list.shape #(28,302)
    for i in range(0,num_cols):
        if i == 0:
            ax.plot(t_list, data_list[:,i], linestyle = 'solid', linewidth = 1.0, \
                    color = col_data, alpha = 0.5, label = 'Simulations') #New data using 302 realz
        else:
            ax.plot(t_list, data_list[:,i], linestyle = 'solid', linewidth = 1.0, \
                    color = col_data, alpha = 0.5) #New data using 302 realz

    ax.plot(t_list, obs_data_list, linestyle = 'solid', linewidth = 1.0, \
            color = 'k', label = 'Observational data') #Obs. data
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = 'best')
    fig.tight_layout()
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')

#**********************************;
# 1. Set new 302 realz data paths  ; 
#**********************************;
path_obs               = "/Users/mudu605/Downloads/Erol_ML_Hydrogeophysics/Original_Outputs/Obs_Potential_Data_2013.txt"
path_new_data          = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/New_Data/"
path_raw_npy           = path_new_data + "302_June10_raw_npy/" #302 raw data realz in .npy format
path_rawmeas_npy       = path_new_data + "302_June10_rawmeas_npy/" #40466 raw measurements in .npy format
path_normmeas_smsd_npy = path_new_data + "302_June10_normmeas_smsd_npy/" #40466 norm measurements by smsd in .npy format
path_norm_smsd_npy     = path_new_data + "302_June10_norm_smsd_npy/" #302 norm data realz by smsd in .npy format
#
path_smsd_plot         = "/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/11_HGP_ML/302_Realz_Plots/"

#***************************************************;
# 2. Plot norm and raw simulations vs observations  ; 
#***************************************************;
num_time_stamps = 28 #Total no. of time-stamps
num_pots        = 40466 #Total no. of potential measurements
num_realz       = 302 #Total no. of new realz
#
pots_list       = list(range(0,num_pots)) #40466 length
realz_list      = list(range(1,num_realz+1)) #302 realizations
#
v_old_obs       = np.genfromtxt(path_obs)[0:num_time_stamps,1:] #(28, 40466)
t_list          = np.genfromtxt(path_obs)[:,0].astype('int64') #time-stamps
#
for pots in range(0,num_pots): #Iterate over 40466 measurements
    col_data      = 'b'
    str_x_label   = r'Time-level [hours]'
    str_y_label   = r'Transfer resistance $\left[\frac{\Delta V}{I} \right]$'
    str_fig_name  = path_smsd_plot + "1_rawvsobs/Sim_rawmeas_" + str(pots) #raw pots-th measurement
    #
    new_meas_data = np.load(path_rawmeas_npy + "Sim_new_" + str(pots) + ".npy")#Each raw pots file (28,302)
    #
    plot_pots_vs_obs(new_meas_data, v_old_obs[:,pots], t_list, pots, \
                     str_x_label, str_y_label, str_fig_name, col_data) #Raw data plots
    #
    y_obs_mean = np.mean(v_old_obs[:,pots]) #Mean of obs data at each pots
    y_obs_std  = np.std(v_old_obs[:,pots]) #STD of obs data at each pots
    vn_old_obs = (v_old_obs[:,pots] - y_obs_mean)/y_obs_std #Norm by measurement for obs data (at each pots)
    #
    col_data           = 'r'
    str_x_label        = r'Time-level [hours]'
    str_y_label        = r'Normalized transfer resistance'
    str_fig_name       = path_smsd_plot + "2_normvsobs/Sim_normmeas_" + str(pots) #normbymeasure smsd pots-th measurement
    #
    new_meas_data_norm = np.load(path_normmeas_smsd_npy + "Sim_new_smsdnorm_" + str(pots) + ".npy") #Each norm pots file (28,302)
    #
    plot_pots_vs_obs(new_meas_data_norm, vn_old_obs, t_list, pots, \
                     str_x_label, str_y_label, str_fig_name, col_data) #Norm data plots
    print('pots = ', pots)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)