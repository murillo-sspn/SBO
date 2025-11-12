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

import os
import time

#-----------------------------------------------------#
# Global variables
#-----------------------------------------------------#

_log_messages = []          # stores messages in memory
_log_file_path = None       # full path to the log file
_t0 = 0     # initial time
flag_print = True

# -----------------------------------------------------#
# Logger
# -----------------------------------------------------#

def init_logger(log_dir="logs", log_filename="log.txt", overwrite=False):
    """Initialize the logger (creates directory, sets file path)."""
    global _log_file_path, _t0

    _t0 = time.time()

    os.makedirs(log_dir, exist_ok=True)
    _log_file_path = os.path.join(log_dir, log_filename)
    if overwrite and os.path.exists(_log_file_path):
        os.remove(_log_file_path)

    log_print(f"Logger initialized: {_log_file_path}")


def set_flag_print(flag_print_new):
    global flag_print
    flag_print = flag_print_new


def get_flag_print():
    global flag_print
    return flag_print


def log_print(message):
    """Print to screen and store message in memory for later saving."""
    global _log_messages

    if flag_print:
        print(message)
        _log_messages.append(format_message(message, time.time() - _t0))


def format_message(message, dt):
    return f"t={'%.2f' % dt}:\t{message}\n"


def save_log():
    """Write all stored log messages to file once."""
    global _log_messages

    log_print(f"Log saved to: {_log_file_path}")

    with open(_log_file_path, "w", encoding="utf-8") as f:
        f.writelines(_log_messages)


def close_logger(final_message="Logger finished."):
    """Convenience wrapper to finalize and save     ."""
    save_log()
    print(final_message)

# -----------------------------------------------------#
# Logger
# -----------------------------------------------------#

def print_banner():
    log_print("#######################################################")
    log_print("#                                                     #")
    log_print("#           Surrogate Based Optimization              #")
    log_print("#     framework for multistage compressor design      #")
    log_print("#                                                     #")
    log_print("#######################################################")
    log_print('')