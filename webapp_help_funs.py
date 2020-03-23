# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:06:40 2020

@author: MattiaL
"""

import os

def mkdir_fun(folder):
    '''
    Creates a folder if it is not there
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)