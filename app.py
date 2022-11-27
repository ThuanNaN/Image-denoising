import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import time

from models import load_model, predict_Restormer, predict_SADNet
from  utils import resize_img

device = "cuda" if torch.cuda.is_available() else "cpu"


Restormer_model = load_model("Restormer")
Restormer_model.to(device)

SADNet_model  = load_model("SADNet")
SADNet_model.to(device)


st.subheader("Image Denoising")

with st.sidebar:
    option = st.selectbox(
            'Choose model: ',
            ('Restormer', 'SADNet')
        )

    st.write('You selected:', option)






    
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    img_array = np.array(Image.open(image_file))


    with st.sidebar:
        resize_choice = st.checkbox("Resize Image")
        if resize_choice:
            scale_values = st.slider('Resize scale (%)', -100, 100, 0 )
            st.write("Scale ratio: {}%".format(scale_values))
            img_array = resize_img(img_array, scale_values)
        
    
    st.image(img_array, width=650)
    st.write("Image shape: {}".format(img_array.shape))

    if st.button("RUN"):
        if option == "Restormer":
            result = predict_Restormer(Restormer_model, img_array, device)

        elif option == "SADNet":
            result = predict_SADNet(SADNet_model, img_array, device)


        if result["status"]  == "Success":
            st.success("Success !!!", icon = "✅")
            
        elif result["status"]  == "Fail":
            st.warning('Error !!!', icon="🚨")

        st.image(result["data"],width=650)



