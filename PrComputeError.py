# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 16:50:21 2021

@author: meslami
"""

import pandas
import numpy as np



''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Functions:
'''


def FnConvert_1D_VF_2_2D(Vect54):
    
    VF_2D=np.zeros(shape=(8,9),dtype=int)
    VF_2D=VF_2D-3
    
    VF_2D[0,3:7]=Vect54[0:4]
    VF_2D[1,2:8]=Vect54[4:10]
    VF_2D[2,1:9]=Vect54[10:18]
    VF_2D[3,0:9]=Vect54[18:27]
    VF_2D[4,0:9]=Vect54[27:36]
    VF_2D[5,1:9]=Vect54[36:44]
    VF_2D[6,2:8]=Vect54[44:50]
    VF_2D[7,3:7]=Vect54[50:54]
    
    return VF_2D


''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Settings: 
'''

InputCSV_Truth='./Competitor1/Outputs/VF_truth_Fold_1.csv'
InputCSV_Pred='./Competitor1/Outputs/VF_pred_Fold_1.csv'
#OutputFolder='./Outputs/'

VFs_truth=pandas.read_csv(InputCSV_Truth, low_memory=False)
VFs_pred=pandas.read_csv(InputCSV_Pred, low_memory=False)


PMAE=np.mean(np.absolute(VFs_truth.values-VFs_pred.values), axis=1)


import matplotlib.pyplot as plt

for k in range(len(VFs_truth)):
    VF_truth=VFs_truth.iloc[k].values
    VF_pred=VFs_pred.iloc[k].values
        
#    plt.figure()
#    plt.subplot(122)
#    plt.imshow(np.reshape(VF_truth,(8,9)))
#    #plt.colorbar()
#    plt.title('Truth')
#    plt.subplot(121)
#    plt.imshow(np.reshape(VF_pred,(8,9)))
#    plt.title('Pred')
    #plt.save()
    #plt.close()
    
    