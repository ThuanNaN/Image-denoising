import wget
import os
import gdown
import cv2

def download_RestormerW():
    path_save = os.path.join("Restormer",'Denoising', 'pretrained_models')
    url =  "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
    wget.download(url, out = path_save)


def download_SADW():
    w_ID = "10HdJeTwvcJ804lQOZPk4fMLJEQaJx8Yc"
    gdown.download(id=w_ID, output = "./SADNet/")


def resize_img(image, ratio):
    h, w, _ = image.shape

    scale = int(100 + ratio + 1) / 100

    new_w = int(w*scale)
    new_h = int(h*scale)

    img = cv2.resize(image, (new_w, new_h))

    return img