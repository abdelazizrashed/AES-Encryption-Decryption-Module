# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:11:32 2020

@author: Abdelaziz Rashed
"""
from pyfinite import genericmatrix
from pyfinite import ffield
import AESFunctions
import numpy as np


"""
Generate the S-Box table
"""
#lambda function that define the bitwise oprations
XOR = lambda x,y : x^y
AND = lambda x,y : x&y
DIV = lambda x,y : x

mat = SBoxGenerator()
print(mat)