# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:08:32 2021

@author: HZU
"""
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import numpy as np
import os

##For normal dices
for j in range(0,11):
    folder_path = os.path.abspath('./')
    locationFiles=str(folder_path)+"/normal_dice/"+str(j)
    all_files = os.listdir(locationFiles)
    new_path = os.path.join(folder_path, 'arr_'+str(j))
    os.mkdir(new_path)
    text_files=[]
    for i in range(len(all_files)):
        if all_files[i][-4:]=='.jpg':
            # load the image
            image = Image.open(os.path.join(locationFiles, all_files[i]))
            image_bw = image.convert('L')
            # convert image to numpy array
            data = asarray(image_bw)
            data_fft = abs(fft(data))
            all_files[i] = all_files[i].replace('.jpg', '')
            np.save(os.path.join(new_path, all_files[i]), data_fft)
        else:
            next

# x = np.load('./1001.npy')


##For anomalous_dices

folder_path = os.path.abspath('./')
locationFiles=str(folder_path)+"/anomalous_dice/"
all_files = os.listdir(locationFiles)
new_path = os.path.join(folder_path, 'anomalous_dice_arr')
os.mkdir(new_path)
text_files=[]
for i in range(len(all_files)):
    if all_files[i][-4:]=='.jpg':
        # load the image
        image = Image.open(os.path.join(locationFiles, all_files[i]))
        image_bw = image.convert('L')
        # convert image to numpy array
        data = asarray(image_bw)
        data_fft = abs(fft(data))
        all_files[i] = all_files[i].replace('.jpg', '')
        np.save(os.path.join(new_path, all_files[i]), data_fft)
    else:
        next
