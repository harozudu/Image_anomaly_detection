# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:12:52 2021

@author: HZU
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import os


class_names = [0,1,2,3,4,5,6,7,8,9,10]
class_names_FFT = {0: "One", 1: "Two", 2: "Two", 3: "Three", 4: "Three", 5: "Four",
           6: "Five", 7: "Faktion Logo", 8: "Faktion Logo", 9: "Faktion Logo",
           10: "Faktion Logo"}
img_all_classes = np.load('./utils/img_all_classes.npy')

def plot_image_input(img, prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction)
    plt.gca().set_title('Image to predict')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.xlabel("Label predicted : {}".format(class_names[predicted_label])+"\n"+
               "Similarity value : {:2.0f}".format(np.max(prediction))+"\n")
    return fig

def plot_image_input_FFT(img, detected_class, detected_anomaly):
    fig, ax = plt.subplots()
    plt.gca().set_title('Image to predict')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.xlabel(f"Label predicted : {class_names_FFT[detected_class]}" +"\n" +
               f"Anomaly detection : {str(detected_anomaly).upper()}")
    return fig

def plot_predict_histo(prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction.tolist()[0])
    plt.gca().set_title('Image to predict') 
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    thisplot = ax.bar(class_names, prediction.tolist()[0], color="#777777")
    thisplot[predicted_label].set_color('red')
    return fig

def plot_same_label_img(prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction)
    plt.gca().set_title('Random image with same class') 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(img_all_classes[predicted_label], cmap='gray', vmin=0, vmax=255)
    plt.xlabel("Class value: {}".format(predicted_label))
    return fig

def plot_FFT(fft):

    fig = go.Figure(data=[go.Surface(z=fft)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    return fig
