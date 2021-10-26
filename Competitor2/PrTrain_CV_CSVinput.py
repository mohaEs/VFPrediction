
"""
Created on Fri Feb 26 09:35:14 2021

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



InputCSV_PairsData='../FakeData_CSV_Samples/PairedInfo_right.csv'
InputCSV_VFData='../FakeData_CSV_Samples/Dataset_right.csv'

OutputFolder='./Outputs/'


import os
try:
    os.stat(OutputFolder)
except:
    os.mkdir(OutputFolder) 
    
    

Data_PairedInfo=pandas.read_csv(InputCSV_PairsData, low_memory=False)
Data_VF=pandas.read_csv(InputCSV_VFData, low_memory=False)



''' ######################################## '''
''' ######################################## '''
''' ######################################## 
Filter the data to have at least 6 timesteps
'''

IDs_0=Data_PairedInfo['ID'].unique()

NumRecords=Data_PairedInfo.groupby(['ID']).count()
IDs=NumRecords[NumRecords.Interval>4].index
Data_PairedInfo_filtered=Data_PairedInfo[Data_PairedInfo['ID'].isin(IDs)]
IDs=Data_PairedInfo_filtered['ID'].unique()


''' ######################################## '''
''' ######################################## '''
''' ######################################## '''

ind_vf_start=Data_VF.columns.get_loc("s1")
ind_vf_end=Data_VF.columns.get_loc("s54")

ind_td_start=Data_VF.columns.get_loc("td1")
ind_td_end=Data_VF.columns.get_loc("td54")

ind_pd_start=Data_VF.columns.get_loc("pd1")
ind_pd_end=Data_VF.columns.get_loc("pd54")

ind_falsenegrate=Data_VF.columns.get_loc("falsenegrate")
ind_falseposrate=Data_VF.columns.get_loc("falseposrate")
ind_malfixrate=Data_VF.columns.get_loc("malfixrate")



''' ######################################## '''
''' ######################################## '''
''' ######################################## '''


import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import models
SelectedModel = models.ModelOfPaper()
SelectedModel.summary()
import tensorflow as tf
OPTIMIZER_2=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
SelectedModel.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=OPTIMIZER_2)
SelectedModel.save_weights('SavedInitialWeights_tensors.h5')


''' ######################################## '''
''' ######################################## '''
''' ######################################## '''




import time

kf10 = KFold(n_splits=3, shuffle=True)
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
    X_VFs_tensor_train=np.zeros(shape=(len(Data_PairedInfo_filtered),6,108))
    Y_VFs_tensor_train=np.zeros(shape=(len(Data_PairedInfo_filtered),52))            
    counter_=-1
    for i in range(0,len(train_idx)):#
        idd=IDs[train_idx[i]]
        #print(id)
        
        pair_id_interest=Data_PairedInfo_filtered[Data_PairedInfo_filtered.ID==idd]
        pair_id_interest=pair_id_interest.sort_values(by=['Interval'])                     
        
        for ii in range(0,len(pair_id_interest)-4):            
            counter_=counter_+1
            
            index_input_4_Data_VF_0=pair_id_interest.Index_input.iloc[ii]
            index_input_4_Data_VF_1=pair_id_interest.Index_output.iloc[ii]
            index_input_4_Data_VF_2=pair_id_interest.Index_output.iloc[ii+1]
            index_input_4_Data_VF_3=pair_id_interest.Index_output.iloc[ii+2]
            index_input_4_Data_VF_4=pair_id_interest.Index_output.iloc[ii+3]
            index_output_4_Data_VF=pair_id_interest.Index_output.iloc[ii+4]
            
            time_pred_years=pair_id_interest.Interval.iloc[ii+4]
            time_last_years=pair_id_interest.Interval.iloc[ii+3]
            time_0_days=round((-time_last_years)*365)
            time_1_days=round((pair_id_interest.Interval.iloc[ii]-time_last_years)*365)
            time_2_days=round((pair_id_interest.Interval.iloc[ii+1]-time_last_years)*365)
            time_3_days=round((pair_id_interest.Interval.iloc[ii+2]-time_last_years)*365)
            time_4_days=0
                                
            PDV_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_pd_start:ind_pd_end+1] / 50 
            TDV_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_pd_start:ind_pd_end+1] / 50 
            TDV_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_pd_start:ind_pd_end+1] / 50 
            TDV_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_td_start:ind_td_end+1] / 50
            
            FP_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_falseposrate]
            FN_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_falsenegrate]
            FL_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_malfixrate] 
            FP_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_falseposrate]
            FN_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_falsenegrate]
            FL_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_malfixrate] 
            FP_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_falseposrate]
            FN_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_falsenegrate]
            FL_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_malfixrate] 
            FP_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_falseposrate]
            FN_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_falsenegrate]
            FL_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_malfixrate] 
            FP_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_falseposrate]
            FN_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_falsenegrate]
            FL_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_malfixrate]          
            
            Tmp=np.array([time_0_days, FP_input_0,FN_input_0,FL_input_0])
            Feature_0=np.concatenate((Tmp,PDV_input_0,TDV_input_0), axis=0)
            Tmp=np.array([time_1_days, FP_input_1,FN_input_1,FL_input_1])
            Feature_1=np.concatenate((Tmp,PDV_input_1,TDV_input_1), axis=0)
            Tmp=np.array([time_2_days, FP_input_2,FN_input_2,FL_input_2])
            Feature_2=np.concatenate((Tmp,PDV_input_2,TDV_input_2), axis=0)
            Tmp=np.array([time_3_days, FP_input_3,FN_input_3,FL_input_3])
            Feature_3=np.concatenate((Tmp,PDV_input_3,TDV_input_3), axis=0)
            Tmp=np.array([time_4_days, FP_input_4,FN_input_4,FL_input_4])
            Feature_4=np.concatenate((Tmp,PDV_input_4,TDV_input_4), axis=0)
            Feature_5=np.zeros(shape=Feature_4.shape,dtype=np.float64)
            Feature_5[0]=round((time_pred_years-time_last_years)*365)
            
            
            X_VFs_tensor_train[counter_,0,:]=Feature_0
            X_VFs_tensor_train[counter_,1,:]=Feature_1
            X_VFs_tensor_train[counter_,2,:]=Feature_2
            X_VFs_tensor_train[counter_,3,:]=Feature_3
            X_VFs_tensor_train[counter_,4,:]=Feature_4
            X_VFs_tensor_train[counter_,5,:]=Feature_5
            
            Y_VFs_tensor_train[counter_,:]=Data_VF.values[index_output_4_Data_VF.astype(int),ind_td_start:ind_td_end+1] /50   
            
    X_VFs_tensor_train_del=np.delete(X_VFs_tensor_train, range(counter_+1,len(Data_PairedInfo_filtered)), 0)
    X_VFs_tensor_train_del[:,:,0]=X_VFs_tensor_train_del[:,:,0]/10000
    Y_VFs_tensor_train_del=np.delete(Y_VFs_tensor_train, range(counter_+1,len(Data_PairedInfo_filtered)), 0)     
    print('Number of samples - Training: ', Y_VFs_tensor_train_del.shape[0])

#        plt.imshow(X_VFs_tensor_train_del[29,:,:,0])
#        plt.colorbar()
        
    ''' arrange the test samples ###########
    '''
    X_VFs_tensor_test=np.zeros(shape=(len(Data_PairedInfo_filtered),6,108))
    Y_VFs_tensor_test=np.zeros(shape=(len(Data_PairedInfo_filtered),52))              
    counter_=-1
    for i in range(0,len(test_idx)):#
        idd=IDs[test_idx[i]]
        #print(id)
        
        pair_id_interest=Data_PairedInfo_filtered[Data_PairedInfo_filtered.ID==idd]
        pair_id_interest=pair_id_interest.sort_values(by=['Interval'])  
                
        for ii in range(0,len(pair_id_interest)-4):
            
            counter_=counter_+1

            index_input_4_Data_VF_0=pair_id_interest.Index_input.iloc[ii]
            index_input_4_Data_VF_1=pair_id_interest.Index_output.iloc[ii]
            index_input_4_Data_VF_2=pair_id_interest.Index_output.iloc[ii+1]
            index_input_4_Data_VF_3=pair_id_interest.Index_output.iloc[ii+2]
            index_input_4_Data_VF_4=pair_id_interest.Index_output.iloc[ii+3]
            index_output_4_Data_VF=pair_id_interest.Index_output.iloc[ii+4]
            
            time_pred_years=pair_id_interest.Interval.iloc[ii+4]
            time_last_years=pair_id_interest.Interval.iloc[ii+3]
            time_0_days=round((-time_last_years)*365)
            time_1_days=round((pair_id_interest.Interval.iloc[ii]-time_last_years)*365)
            time_2_days=round((pair_id_interest.Interval.iloc[ii+1]-time_last_years)*365)
            time_3_days=round((pair_id_interest.Interval.iloc[ii+2]-time_last_years)*365)
            time_4_days=0
                                
            PDV_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_td_start:ind_td_end+1] / 50
            PDV_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_pd_start:ind_pd_end+1] / 50
            TDV_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_td_start:ind_td_end+1] / 50
            
            FP_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_falseposrate]
            FN_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_falsenegrate]
            FL_input_0=Data_VF.values[index_input_4_Data_VF_0.astype(int),ind_malfixrate] 
            FP_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_falseposrate]
            FN_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_falsenegrate]
            FL_input_1=Data_VF.values[index_input_4_Data_VF_1.astype(int),ind_malfixrate] 
            FP_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_falseposrate]
            FN_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_falsenegrate]
            FL_input_2=Data_VF.values[index_input_4_Data_VF_2.astype(int),ind_malfixrate] 
            FP_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_falseposrate]
            FN_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_falsenegrate]
            FL_input_3=Data_VF.values[index_input_4_Data_VF_3.astype(int),ind_malfixrate] 
            FP_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_falseposrate]
            FN_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_falsenegrate]
            FL_input_4=Data_VF.values[index_input_4_Data_VF_4.astype(int),ind_malfixrate] 
            
            Tmp=np.array([time_0_days, FP_input_0,FN_input_0,FL_input_0])
            Feature_0=np.concatenate((Tmp,PDV_input_0,TDV_input_0), axis=0)
            Tmp=np.array([time_1_days, FP_input_1,FN_input_1,FL_input_1])
            Feature_1=np.concatenate((Tmp,PDV_input_1,TDV_input_1), axis=0)
            Tmp=np.array([time_2_days, FP_input_2,FN_input_2,FL_input_2])
            Feature_2=np.concatenate((Tmp,PDV_input_2,TDV_input_2), axis=0)
            Tmp=np.array([time_3_days, FP_input_3,FN_input_3,FL_input_3])
            Feature_3=np.concatenate((Tmp,PDV_input_3,TDV_input_3), axis=0)
            Tmp=np.array([time_4_days, FP_input_4,FN_input_4,FL_input_4])
            Feature_4=np.concatenate((Tmp,PDV_input_4,TDV_input_4), axis=0)
            Feature_5=np.zeros(shape=Feature_4.shape,dtype=np.float64)
            Feature_5[0]=round((time_pred_years-time_last_years)*365)
            
            
            X_VFs_tensor_test[counter_,0,:]=Feature_0
            X_VFs_tensor_test[counter_,1,:]=Feature_1
            X_VFs_tensor_test[counter_,2,:]=Feature_2
            X_VFs_tensor_test[counter_,3,:]=Feature_3
            X_VFs_tensor_test[counter_,4,:]=Feature_4
            X_VFs_tensor_test[counter_,5,:]=Feature_5
            
            Y_VFs_tensor_test[counter_,:]=Data_VF.values[index_output_4_Data_VF.astype(int),ind_td_start:ind_td_end+1] /50     

            
    X_VFs_tensor_test_del=np.delete(X_VFs_tensor_test, range(counter_+1,len(Data_PairedInfo_filtered)), 0)
    X_VFs_tensor_test_del[:,:,0]=X_VFs_tensor_test_del[:,:,0]/10000
    Y_VFs_tensor_test_del=np.delete(Y_VFs_tensor_test, range(counter_+1,len(Data_PairedInfo_filtered)), 0)        
    
    
#    plt.imshow(Y_VFs_tensor_test_del[0,:,:,0])
#    plt.colorbar()        
    

    ''' Training ###########
    '''  
    print('Training started ...')  
    start_time = time.time()     
    History = SelectedModel.fit(X_VFs_tensor_train_del, Y_VFs_tensor_train_del, validation_split=0.1, epochs=300, batch_size=2, verbose=0)#250-250
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
    #Y_pred_VFs_tensor_train_del=SelectedModel.predict(X_VFs_tensor_train_del)
    
#    plt.imshow(Y_pred_VFs_tensor_test_del[0,:,:,0])
#    plt.colorbar() 
    
    
    Y_VFs_tensor_test_del=50*Y_VFs_tensor_test_del
    Y_pred_VFs_tensor_test_del=50*Y_pred_VFs_tensor_test_del
    
           
    NameTruth=OutputFolder+'TDV_Fold_' +str(fold_counter) +'_truth.csv' 
    NamePred=OutputFolder+'TDV_Fold_' + str(fold_counter) +'_pred.csv'
            
    np.savetxt(NameTruth, Y_VFs_tensor_test_del, delimiter=',', fmt='%1.3f')   
    np.savetxt(NamePred, Y_pred_VFs_tensor_test_del, delimiter=',', fmt='%1.3f') 
    
        
        