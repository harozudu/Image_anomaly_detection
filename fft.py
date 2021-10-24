# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:26:51 2021

@author: HZU
"""
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import pandas as pd
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go

# load the image
image_defect = Image.open('img_17413_cropped.jpg')
image_normal = Image.open('674.jpg')
image_defect_bw = image_defect.convert('L')
image_normal_bw = image_normal.convert('L')

# convert image to numpy array
data_defect = asarray(image_defect_bw)
data_normal = asarray(image_normal_bw)

data_defect_fft = abs(fft(data_defect))
data_normal_fft = abs(fft(data_normal))


clipped_df = pd.DataFrame(abs(data_normal_fft))

fig = go.Figure(data=[go.Surface(z=clipped_df)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

fig.show()