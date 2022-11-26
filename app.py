import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from models import load_Restormer, predict_Restormer



model = load_Restormer()


def load_image(image_file):
	img = Image.open(image_file)
	return img

choice = "Image"

if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

        img_array = np.array(Image.open(image_file))

        result_img = predict_Restormer(model, img_array, device = "cpu")


        # # To See details
        # file_details = {"filename":image_file.name, "filetype":image_file.type,
        #                 "filesize":image_file.size}
        # st.write(file_details)

        # To View Uploaded Image
        st.image(load_image(result_img),width=250)



