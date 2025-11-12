#######################################################
#                                                     #
#           Surrogate Based Optimization              #
#     framework for multistage compressor design      #
#                                                     #
#######################################################
# Authors:                                            #
#    Murillo S. S. Pereira Neto,                      #
#    @ University of São Paulo, Brazil                #
#    Bruno José Almeida Nagy,                         #
#    @ University of São Paulo, Brazil                #
#                                                     #
#                                                     #
# Description:                                        #
#######################################################
import shutil
#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import sys
import os
import pandas as pd
import numpy as np

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.logger import *


class Directories():

    def __init__(self, cfg_filepath):

        self.cfg_filepath           = cfg_filepath
        self.inputs_directory       = os.path.dirname(self.cfg_filepath)
        self.case_directory         = os.path.dirname(self.inputs_directory)
        self.case_name              = os.path.basename(self.case_directory)
        self.outputs_directory      = os.path.join(self.case_directory, 'Outputs')
        shutil.rmtree(self.outputs_directory, ignore_errors=True)
        os.makedirs(self.outputs_directory, exist_ok=True)

def csv_to_dict(filename):
    # Load the header (first row)
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')

    # Load the remaining numeric data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Build a dictionary: {column_name: numpy_array}
    data_dict = {header[i]: np.array([data[:, i]]) for i in range(len(header))}

    return data_dict



