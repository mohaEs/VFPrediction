

import pandas
import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


InputFolder='../FakeData_Images_Samples/'
OutputFolder='./OutputModel/'

import os
try:
    os.stat(OutputFolder)
except:
    os.mkdir(OutputFolder) 



X_all_names=[]

import glob
import skimage
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize

for filename in glob.glob(InputFolder+'/*.png'): #assuming png
    #FileName=filename.replace(SourcePath,'')
    #print(filename)    
    X_all_names.append(filename)


X_all=np.zeros((len(X_all_names),8,9,1),dtype=np.float32)
for i in range(len(X_all_names)): #assuming png
    #FileName=filename.replace(SourcePath,'')
    #print(filename)    
    IMG = io.imread(X_all_names[i])
#    print(IMG.shape)    
    IMG_gray=skimage.color.rgb2grey(IMG)
#    plt.imshow(IMG_gray)
#    print(IMG_gray.shape)    
    IMG_gray_sized=resize(IMG_gray,(8,9))
    #print(IMG_gray_sized.shape)
    #plt.imshow(IMG_gray_sized)        
    X_all[i,:,:,0]=IMG_gray_sized    
plt.imshow(X_all[5,:,:,0])    
plt.close()


import models
SelectedModel = models.getModel("CascadeNet-5", (8,9,1))
SelectedModel.summary()

callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

Y_all=X_all

import tensorflow as tf
# OPTIMIZER_2=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# SelectedModel.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=OPTIMIZER_2)

History = SelectedModel.fit(X_all, Y_all, validation_split=0.1,  epochs=300, batch_size=1,  callbacks=[callback_stop], verbose=1)

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

