
import torch
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

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
    input_img = Image.open(image_file)
    img_array = np.array(input_img)


    with st.sidebar:

        crop_choice = st.checkbox("Crop Image")
        if crop_choice:
            aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9","Free"])
            aspect_dict = {
                "1:1": (1, 1),
                "Free": None
            }
            aspect_ratio = aspect_dict[aspect_choice]

            rect = st_cropper(
                Image.fromarray(img_array),
                realtime_update=True,

                aspect_ratio=aspect_ratio,
                return_type='box'
            )
            left, top, width, height = tuple(map(int, rect.values()))
            img_array = img_array[top:top+height, left:left+width]


        resize_choice = st.checkbox("Resize Image")
        if resize_choice:
            scale_values = st.slider('Resize scale (%)', -200, 200, 0 )
            st.write("Scale ratio: {}%".format(scale_values))
            img_array = resize_img(img_array, scale_values)

        

        compare_choice = st.checkbox("Compare Result")


    
    st.image(img_array, width=500)
    st.write("Image shape: {}".format(img_array.shape))

    if st.button("RUN"):
        
        if compare_choice:
            col1, col2 = st.columns(2, gap="small")
            with col1:
                st.header("Restormer")
                result = predict_Restormer(Restormer_model, img_array, device)

                if result["status"]  == "Success":
                    st.success("Success !!!", icon = "✅")
                    
                elif result["status"]  == "Fail":
                    st.warning('Error !!!', icon="🚨")

                st.image(result["data"],width=300)
            
            with col2: 
                st.header("SADNet")
                result = predict_SADNet(SADNet_model, img_array, device)
                
                if result["status"]  == "Success":
                    st.success("Success !!!", icon = "✅")
                    
                elif result["status"]  == "Fail":
                    st.warning('Error !!!', icon="🚨")

                st.image(result["data"],width=300)


        else:
            if option == "Restormer":
                result = predict_Restormer(Restormer_model, img_array, device)

            elif option == "SADNet":
                result = predict_SADNet(SADNet_model, img_array, device)

            if result["status"]  == "Success":
                st.success("Success !!!", icon = "✅")
                
            elif result["status"]  == "Fail":
                st.warning('Error !!!', icon="🚨")

            st.image(result["data"],width=500)



