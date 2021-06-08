# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:04:47 2021

@author: harsh
"""

import pickle
import numpy as np

local_classifier=pickle.load(open('classifier.pickle','rb'))
local_scaler=pickle.load(open('sc.pickle','rb'))

