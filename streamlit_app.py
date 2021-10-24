# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io
import joblib
from utils.plot_functions import plot_image_input_FFT, plot_image_input, \
    plot_predict_histo, plot_same_label_img, plot_FFT
from fft_2 import preprocess_input, fft_detector
import torch
import torchvision.transforms as tt

st.set_page_config(layout="wide")

st.header('Faktion-anomaly-detection')
st.subheader('Anomaly detection in images')

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
##Loading the image to be analize
st.write('Please upload a dice image 128x128')
image_file = st.file_uploader("Upload File", type=['jpg'])

if image_file:
    file_details = {"FileName": image_file.name, "FileType": image_file.type,
                    "FileSize": image_file.size}
    st.image(image_file)
    st.write(file_details)
    data = image_file.read()
    dataBytesIO = io.BytesIO(data)
    pil_img = Image.open(dataBytesIO)
    img = np.array(pil_img)

# Loading the model and also all the classes images to plot
model = tf.keras.models.load_model('./utils/cnn_model.h5')

# One image for each class is loaded here named from 0 to 10
img_all_classes = np.load('./utils/img_all_classes.npy')
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
value_to_use_if = None

with st.expander('CNN method'):
    col_mid, col1, col2, col3, col_end = st.columns(5)
    if image_file:
        prediction = model.predict(img[None, :, :])
        plot_1 = plot_image_input(img, prediction)
        plot_2 = plot_predict_histo(prediction)
        plot_3 = plot_same_label_img(prediction)
        col1.pyplot(plot_1)
        col2.pyplot(plot_2)
        col3.pyplot(plot_3)

with st.expander('FFT method'):
    col_mid, col_left, col_mid, col_right, col_mid = st.columns(
        (0.7, 1, 0.7, 1, 0.7))

    # To play with between 0 (Everything is an anomaly) and 1 (No False Positives on Training)
    predictive_strength = col_right.slider("Precision-Recall Trade-Off",
                                           min_value=0.0, max_value=1.0,
                                           value=1.0, step=0.05)

    if image_file:
        processed_image = preprocess_input(
            img)  # Apply the preprocessing on the matrix image
        detected_anomaly, detected_class, false_positives_on_training_set = fft_detector(
            processed_image, predictive_strength)

        plot_FFT = plot_image_input_FFT(img, detected_class, detected_anomaly)
        col_left.pyplot(plot_FFT)

        col_right.write(
            "The trade-off between recall and precision generated the following false positives on the Training Set")
        col_right.write(false_positives_on_training_set)
        value_to_use_if = 1

    st.markdown("""---""")

    col_left, col_right = st.columns((1, 1))

    with col_left:
        if value_to_use_if:
            fig = go.Figure(data=[go.Surface(z=processed_image)])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                              highlightcolor="limegreen",
                                              project_z=True))
            fig.update_layout(title={'text': "FFT for Image input"})
            fig.update_layout(width=800,
                              height=800)
            st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with col_right:
        if value_to_use_if:
            models = joblib.load("utils/models.pkl")
            fig = go.Figure(data=[go.Surface(z=models[detected_class])])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                              highlightcolor="limegreen",
                                              project_z=True))
            fig.update_layout(title={'text': "FFT for class template"})
            fig.update_layout(width=800,
                              height=800)
            st.plotly_chart(figure_or_data=fig, use_container_width=True)


@st.cache
def load_discriminator():
    return torch.load(
        "model/discriminator.pt",
        map_location=torch.device('cpu')
    )


with st.expander('DCGAN Anomaly detection'):
    st.title("Deep Convolutional Generative Adversial Network")
    st.write("Use a DCGAN "
             "trained on clean Dice to detect defective dice.")
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    tt_transform = tt.Compose([tt.Resize(128),
                               tt.CenterCrop(128),
                               tt.ToTensor(),
                               tt.Normalize(*stats)])
    disc = load_discriminator()
    if image_file:
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        gan_img = pil_img.convert('RGB')
        img_tensor = tt_transform(gan_img).unsqueeze_(0)
        score = disc(img_tensor)[0].item()
        score = round(score, 2)
        predict = "Normal" if score >= threshold else "Defective"
        st.write(f"Prediction: {predict}")
        st.write(f"Score: {score}")

