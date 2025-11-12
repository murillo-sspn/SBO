#!/usr/bin/env python3

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


#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import sys
import os

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
sys.path.append(os.environ["SBO_HOME"])
from common.config import *
from common.logger import *
from src.driver import *

class SBO():
    def __init__(self):
        # -----------------------------------------------------#
        # Initialization
        # -----------------------------------------------------#
        # Initializing the DIR and load the configuration file
        self.DIR = os.getcwd() + '/'
        self.INFilename = sys.argv[-1]
        self.INFile = self.DIR + self.INFilename

        # Set directories structure
        self.directories = Directories(self.INFile)

        # Initialize logger
        init_logger(log_dir=self.directories.outputs_directory)

        # Print Banner
        print_banner()

        # Read inputs file
        self.IN = read_user_input(self.INFile)

        # -----------------------------------------------------#
        # Run
        # -----------------------------------------------------#

        driver = Driver(self.IN, self.directories)

        breakpoint()

        # -----------------------------------------------------#
        # Finalization
        # -----------------------------------------------------#

        # Print Banner
        close_logger()



def main():
    sbo = SBO()

if __name__ == '__main__':
    main()