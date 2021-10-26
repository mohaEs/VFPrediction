"""
Created on Fri Jan 29 20:07:22 2021

@author: meslami
"""

'''
Convert 1D 24-2 Visual field data to 2D matrix
'''


import numpy as np
VFs=np.array(range(1,55))

VF_2D=np.zeros(shape=(8,9),dtype=int)
VF_2D=VF_2D-3

VF_2D[0,3:7]=VFs[0:4]
VF_2D[1,2:8]=VFs[4:10]
VF_2D[2,1:9]=VFs[10:18]
VF_2D[3,0:9]=VFs[18:27]
VF_2D[4,0:9]=VFs[27:36]
VF_2D[5,1:9]=VFs[36:44]
VF_2D[6,2:8]=VFs[44:50]
VF_2D[7,3:7]=VFs[50:54]

