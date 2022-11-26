import wget
import os

def download_RestormerW(weights_path):
    path_save = os.path.join("Restormer",'Denoising', 'pretrained_models')
    url =  "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
    weight = wget.download(url, out = path_save)