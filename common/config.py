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
import re
import numpy as np

def strip_list(lst):

    """ If `lst` is a nested Python list with length 1, the outermost list is stripped

    Parameters
    ----------
    lst : list
        Python list to be stripped

    Returns
    -------
    list

    """

    return lst[0] if (isinstance(lst, list) and len(lst) == 1) else lst

def parse_value(value):

    """ Parses a value of a configuration file

    `;` Delimits values between zones
    `&` Delimits the multiple objectives or constraints per zone

    Delimited values are put into lists

    Values are parsed into floats or integers where possible
    'YES', 'NO', 'NONE' are parsed into True, False, None respectively

    Parameters
    ----------
    value : str
        String of the value to be parsed

    Returns
    -------
    any type
        Parsed `value`

    """

    # Zone delimiter
    if ';' in value:
        return {
            iZone: strip_list(parse_value(val_i))
            for iZone, val_i in enumerate(value.split(';'))
        }

    # Per-zone objective or constraint delimiter
    if '&' in value:
        return [strip_list(parse_value(val_i)) for val_i in value.split('&')]

    # Remove parentheses
    value = value.strip('( )')

    # split across comma and space
    value = re.split(', |,| ', value)

    # Convert to integer or float if possible
    for i, value_i in enumerate(value):

        # Remove possible quotation marks
        value_i = value_i.strip("'")

        try:
            value[i] = int(value_i)  # Convert to integer if possible
        except ValueError:
            try:
                value[i] = float(value_i)  # Convert to float if possible
            except ValueError:
                pass  # Value is alphabetic, keep as string

    # Parse singular string values
    if len(value) == 1 and isinstance(value[0], str):
        # Convert to boolean or None if possible
        value = {'YES': True, 'NO': False, 'NONE': None, 'False': False, 'True':  True}\
            .get(value[0], value[0])

        # Convert empty string to empty list
        if value == '':
            value = []

    # Convert single strings to lists
    if isinstance(value, str):
        value = [value]

    # Parse lists of lists
    if isinstance(value, list):
        if isinstance(value[0], str):
            if '[[' in value[0] and ']]' in value[-1]:
                # Parsing arguments
                j = 0
                value_ = [[value[0].replace('[[','')]]
                for i, elem in enumerate(value[1:-1]):
                    if isinstance(elem, int) or isinstance(elem, float):
                        value_[j].append(elem)
                    elif isinstance(elem, str):
                        if ']' in elem:
                            value_[j].append(elem.replace(']',''))
                        elif '[' in elem:
                            j += 1
                            value_.append([elem.replace('[', '')])
                        else:
                            value_[j].append(elem)
                value_[j].append(value[-1].replace(']]',''))

                # Convert to floats and integers if possible
                for i, line in enumerate(value_):
                    for j, elem in enumerate(line):
                        if isinstance(elem, str):
                            if elem.isdigit():
                                f = float(elem)
                                if np.abs(f % 1) < 1e-8:
                                    value_[i][j] = int(f)
                                else:
                                    value_[i][j] = f

                # Update value
                value = value_

    return value

def read_user_input(file):

    """ Reads `file` into a dictionary

    `file` Must be a configuration file

    Removes all empty and commented lines and lets the function `parse_value` parse the
    values of the lines in `file`

    Parameters
    ----------
    file : str
        Full path of file to be read

    Returns
    -------
    IN : dict
        Dictionary containing all parsed elements of `file`

    """

    IN = {}
    with open(file, 'r') as infile:
        for line in infile:
            # Remove line returns
            line = line.strip('\r\n')

            if len(line)>0:

                if line[0] in ['[']:
                    section = line[1:-1]
                    IN[section]={}
                else:
                    # Keep only lines with useful data
                    if ("=" not in line) or (line[0] in ['%', '#']):
                        continue

                    # split line across equals sign
                    line = line.split("=", 1)
                    key = line[0].strip()
                    value = line[1].strip()
                    IN[section][key] = parse_value(value)

    return IN