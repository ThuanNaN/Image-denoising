import os
from runpy import run_path
from glob import glob
import cv2
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

from utils import download_RestormerW


def load_Restormer():

    weights = os.path.join("Restormer",'Denoising', 'pretrained_models', 'real_denoising.pth')
    if not os.path.exists(weights):
        download_RestormerW(weights)

    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    parameters['LayerNorm_type'] =  'BiasFree'

    load_arch = run_path(os.path.join("Restormer",'basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    
    return model


def predict_Restormer(model, image, device):
    img_multiple_of = 8
    if device == "cuda":
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_ = torch.from_numpy(image).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

    # Pad the input if not_multiple_of 8
    h,w = input_.shape[2], input_.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

    try:
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        return {
                "data": restored,
                "status": "Succes"
                }
    except:
        return {
                "data": image,
                "status": "Fail"
                }