
"""
Created on Fri Jan 29 20:07:22 2021

@author: meslami
"""

import pandas
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Functions:
'''


def FnConvert_1D_VF_2_2D(Vect54):
    
    VF_2D=np.zeros(shape=(8,9),dtype=int)
    
    # set background to -3, since our minimum sensitivity value is -2:    
    VF_2D = VF_2D-3 
    
    VF_2D[0,3:7]=Vect54[0:4]
    VF_2D[1,2:8]=Vect54[4:10]
    VF_2D[2,1:9]=Vect54[10:18]
    VF_2D[3,0:9]=Vect54[18:27]
    VF_2D[4,0:9]=Vect54[27:36]
    VF_2D[5,1:9]=Vect54[36:44]
    VF_2D[6,2:8]=Vect54[44:50]
    VF_2D[7,3:7]=Vect54[50:54]
    
    # since the loss of the paper's method is based on -1 as the background value: 
    VF_2D = VF_2D +2 
    return VF_2D


''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Settings: 
'''



InputCSV_PairsData='../FakeData_CSV_Samples/PairedInfo_right.csv'
InputCSV_VFData='../FakeData_CSV_Samples/Dataset_right.csv'

OutputFolder='./Outputs_y125to175/'
interval_lowerlimit=1.25
interval_higherlimit=1.75

n_epochs_max=500 #paper: 1000 
batch_size=2 #paper: 32


import os
try:
    os.stat(OutputFolder)
except:
    os.mkdir(OutputFolder) 
    
    

Data_PairedInfo=pandas.read_csv(InputCSV_PairsData, low_memory=False)
Data_VF=pandas.read_csv(InputCSV_VFData, low_memory=False)

Data_VF_values = Data_VF.values
Data_VF_values_sensitivities = Data_VF_values[:,36:36+54]

''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Filter the data for specific intervals, e.g. 1.25 - 1.75 years
'''

Data_PairedInfo_FilteredbyYear=Data_PairedInfo[Data_PairedInfo['Interval']<interval_higherlimit]
Data_PairedInfo_FilteredbyYear=Data_PairedInfo_FilteredbyYear[Data_PairedInfo_FilteredbyYear['Interval']>=interval_lowerlimit]


''' ######################################## '''
''' ######################################## '''
''' ######################################## 

'''


IDs=Data_PairedInfo_FilteredbyYear['ID'].unique()
print('==> number of subjects: ', len(IDs))


import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
import models
SelectedModel = models.getModel("CascadeNet-5", (8,9,2))
SelectedModel.summary()
SelectedModel.save_weights('SavedInitialWeights_tensors.h5')

callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

import time

kf10 = KFold(n_splits=10, shuffle=True)
fold_counter=0
for train_idx, test_idx in kf10.split(IDs):
    fold_counter=fold_counter+1
    print('Fold : --------------- ', fold_counter)
    #print(train_idx, test_IDs)
    
    
#    VF_input_2Darray_list=[]
#    VF_output_2Darray_list=[]
    
    SelectedModel.load_weights('SavedInitialWeights_tensors.h5') 
    
    ''' arrange the train samples ###########
    '''
    print('preparing data ...')
    X_VFs_tensor_train=np.zeros(shape=(len(Data_PairedInfo_FilteredbyYear),8,9,2))
    Y_VFs_tensor_train=np.zeros(shape=(len(Data_PairedInfo_FilteredbyYear),8,9,1))            
    counter_=-1
    for i in range(0,len(train_idx)):#
        idd=IDs[train_idx[i]]
        #print(id)
        
        pair_id_interest=Data_PairedInfo_FilteredbyYear[Data_PairedInfo_FilteredbyYear.ID==idd]
        
        for ii in range(0, len(pair_id_interest)):
            
            counter_=counter_+1
            index_input_4_Data_VF=pair_id_interest.Index_input.iloc[ii]
            index_output_4_Data_VF=pair_id_interest.Index_output.iloc[ii]
            
            VF_input=Data_VF_values_sensitivities[index_input_4_Data_VF.astype(int)]
            VF_output=Data_VF_values_sensitivities[index_output_4_Data_VF.astype(int)]
            
            VF_input_2Darray=FnConvert_1D_VF_2_2D(VF_input)
            VF_output_2Darray=FnConvert_1D_VF_2_2D(VF_output)            
#            plt.imshow(VF_input_2Darray)
#            plt.colorbar()
            
#            VF_input_2Darray_list.append(VF_input_2Darray)
#            VF_output_2Darray_list.append(VF_output_2Darray)
#            print('VF_input_2Darray_list',len(VF_input_2Darray_list))
#            print('VF_output_2Darray_list',len(VF_output_2Darray_list))
            
            X_VFs_tensor_train[counter_,:,:,0]=VF_input_2Darray
            Y_VFs_tensor_train[counter_,:,:,0]=VF_output_2Darray
            X_VFs_tensor_train[counter_,:,:,1]=Data_VF['age'].values[index_input_4_Data_VF.astype(int)]
            
    X_VFs_tensor_train_del=np.delete(X_VFs_tensor_train, range(counter_+1,len(Data_PairedInfo_FilteredbyYear)), 0)
    Y_VFs_tensor_train_del=np.delete(Y_VFs_tensor_train, range(counter_+1,len(Data_PairedInfo_FilteredbyYear)), 0)     
    print('Number of samples - Training: ', Y_VFs_tensor_train_del.shape[0])

#        plt.imshow(X_VFs_tensor_train_del[29,:,:,0])
#        plt.colorbar()
        
    ''' arrange the test samples ###########
    '''
    X_VFs_tensor_test=np.zeros(shape=(len(Data_PairedInfo_FilteredbyYear),8,9,2))
    Y_VFs_tensor_test=np.zeros(shape=(len(Data_PairedInfo_FilteredbyYear),8,9,1))            
    counter_=-1
    for i in range(0,len(test_idx)):#
        idd=IDs[test_idx[i]]
        #print(id)
        
        pair_id_interest=Data_PairedInfo_FilteredbyYear[Data_PairedInfo_FilteredbyYear.ID==idd]
        
        for ii in range(0,len(pair_id_interest)):
            
            counter_=counter_+1
            index_input_4_Data_VF=pair_id_interest.Index_input.iloc[ii]
            index_output_4_Data_VF=pair_id_interest.Index_output.iloc[ii]
            
            VF_input=Data_VF.values[index_input_4_Data_VF.astype(int),36:36+54]
            VF_output=Data_VF.values[index_output_4_Data_VF.astype(int),36:36+54]
            
            VF_input_2Darray=FnConvert_1D_VF_2_2D(VF_input)
            VF_output_2Darray=FnConvert_1D_VF_2_2D(VF_output)            
#            plt.imshow(VF_input_2Darray)
#            plt.colorbar()
            
#            VF_input_2Darray_list.append(VF_input_2Darray)
#            VF_output_2Darray_list.append(VF_output_2Darray)
#            print('VF_input_2Darray_list',len(VF_input_2Darray_list))
#            print('VF_output_2Darray_list',len(VF_output_2Darray_list))
            
            X_VFs_tensor_test[counter_,:,:,0]=VF_input_2Darray
            Y_VFs_tensor_test[counter_,:,:,0]=VF_output_2Darray
            X_VFs_tensor_test[counter_,:,:,1]=Data_VF['age'].values[index_input_4_Data_VF.astype(int)]
            
    X_VFs_tensor_test_del=np.delete(X_VFs_tensor_test, range(counter_+1,len(Data_PairedInfo_FilteredbyYear)), 0)
    Y_VFs_tensor_test_del=np.delete(Y_VFs_tensor_test, range(counter_+1,len(Data_PairedInfo_FilteredbyYear)), 0)        
#    plt.imshow(Y_VFs_tensor_test_del[0,:,:,0])
#    plt.colorbar()        
    

    ''' Training ###########
    '''  
    print('Training started ...')  
    start_time = time.time()
    History = SelectedModel.fit(X_VFs_tensor_train_del, Y_VFs_tensor_train_del, validation_split=0.1,
      epochs=n_epochs_max, batch_size=batch_size, callbacks=[callback_stop], verbose=0)#250-250
    elapsed_time = time.time() - start_time
    print('----- train elapsed time:', elapsed_time)
    
    # summarize history for loss
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(OutputFolder+'Fold_'+str(fold_counter)+'_History.png')
    plt.close()

    ''' Testing ###########
    '''  
    
    
    print('Number of samples - Testing: ', Y_VFs_tensor_test_del.shape[0])
    print('Testing started ...')
    Y_pred_VFs_tensor_test_del=SelectedModel.predict(X_VFs_tensor_test_del)
    
#    plt.imshow(Y_pred_VFs_tensor_test_del[0,:,:,0])
#    plt.colorbar() 
    
    VF_input_1D=np.zeros(shape=(X_VFs_tensor_test_del.shape[0],72),dtype=np.float)
    VF_truth_1D=np.zeros(shape=(Y_VFs_tensor_test_del.shape[0],72),dtype=np.float)
    VF_pred_1D=np.zeros(shape=(Y_pred_VFs_tensor_test_del.shape[0],72),dtype=np.float)
        
    for kk in range(0,Y_pred_VFs_tensor_test_del.shape[0]):
        
        VF_truth=Y_VFs_tensor_test_del[kk,:,:,0]
        VF_pred=Y_pred_VFs_tensor_test_del[kk,:,:,0]
        VF_input=X_VFs_tensor_test_del[kk,:,:,0]
        
        VF_pred_1D[kk,::]=VF_pred.reshape((1,72))
        VF_truth_1D[kk,:]=VF_truth.reshape((1,72))
        VF_input_1D[kk,:]=VF_input.reshape((1,72))
        
    NameTruth=OutputFolder+'VF_Fold_' +str(fold_counter) +'_truth.csv' 
    NamePred=OutputFolder+'VF_Fold_' + str(fold_counter) +'_pred.csv'
    NameInput=OutputFolder+'VF_Fold_' + str(fold_counter) +'_input.csv'
        
    
    np.savetxt(NameTruth, VF_truth_1D, delimiter=',', fmt='%1.3f')   
    np.savetxt(NamePred, VF_pred_1D, delimiter=',', fmt='%1.3f') 
    np.savetxt(NameInput, VF_input_1D, delimiter=',', fmt='%1.3f') 
        
        