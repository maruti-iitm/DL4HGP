# DL training at scale -- Embarasingly parallel
# (Create directories on PINKLADY)
# AUTHOR: Maruti Kumar Mudunuru

import numpy as np
import glob
import os
import time
import subprocess
import itertools

if __name__ == '__main__':

    path     = "/home/mudu605/3_HGP_ML/m2_302realz/"
    dir_path = path + "2_m2_PreMixup/"
    print(dir_path)
    #
    if not os.path.exists(dir_path): #Create if they dont exist
        os.makedirs(dir_path)

    random_seed_list = [1337, 0, 1, 2, 3, 4, 5, 7, 8, 10, \
                        11, 13, 42, 100, 113, 123, 200, \
                        1014, 1234, 1410, 1999, 12345, 31337, \
                        141079, 380843] #25 popular random seeds
    num_random_seeds = 15 #Total number of random seeds to be used
    #
    for i in range(0,num_random_seeds):
        random_seed = random_seed_list[i]
        dir_path    = path + "2_m2_PreMixup/" + str(random_seed) + "_model/"
        print(dir_path)
        #
        if not os.path.exists(dir_path): #Create if they dont exist
            os.makedirs(dir_path)