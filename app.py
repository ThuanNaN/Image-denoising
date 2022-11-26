import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from models import load_Restormer, predict_Restormer


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = load_Restormer()
model.to(device)



choice = "Image"

if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

        

        img_array = np.array(Image.open(image_file))
        img_array = cv2.resize(img_array, (100,100))

        st.image(img_array,width=500)

        result = predict_Restormer(model, img_array, device)

        st.write(
            {
                "Status": result["status"]
            })

        st.image(result["data"],width=500)



