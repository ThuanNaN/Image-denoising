import os
from runpy import run_path
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from SADNet.model.__init__ import make_model
import cv2
import numpy as np



from utils import download_RestormerW


def load_model(model_name):
    if model_name == "Restormer":

        weights = os.path.join("Restormer",'Denoising', 'pretrained_models', 'real_denoising.pth')
        if not os.path.exists(weights):
            download_RestormerW()

        parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
        parameters['LayerNorm_type'] =  'BiasFree'

        load_arch = run_path(os.path.join("Restormer",'basicsr', 'models', 'archs', 'restormer_arch.py'))
        model = load_arch['Restormer'](**parameters)

        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['params'])
        
        return model
    elif model_name == "SADNet":
        input_channel, output_channel = 3, 3
        model = make_model(input_channel, output_channel)
        if torch.cuda.is_available():
            model_dict = torch.load("./SADNet/ckpt/SADNET_color_sig50/"+'model_%04d_dict.pth' % 200)
            model.load_state_dict(model_dict)
            model = model.cuda()
        else:
            print('There are not available cuda devices !')

        model.eval()
        return model



def predict_Restormer(model, image, device):
    try:
        img_multiple_of = 8
        if device != "cpu":
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        input_ = torch.from_numpy(image).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
    
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


def predict_SADNet(model, image, device):

    try:

        noisy = image.astype(np.float32) / 255
        noisy = noisy[0:(noisy.shape[0]//8)*8, 0:(noisy.shape[1]//8)*8]
        img_test = transforms.functional.to_tensor(noisy)
        img_test = img_test.unsqueeze_(0).float()
        img_test = img_test.to(device)
        
        torch.cuda.synchronize()
        with torch.no_grad():
            out_image = model(img_test)
        torch.cuda.synchronize()

        rgb_img = out_image.cpu().detach().numpy().transpose((0,2,3,1))
        if noisy.ndim == 3:
            rgb_img = np.clip(rgb_img[0], 0, 1)
        elif noisy.ndim == 2:
            rgb_img = np.clip(rgb_img[0, :, :, 0], 0, 1)

        result_img =  np.uint8(rgb_img*255)

        return {
            "data": result_img,
            "status": "Succes"
            }

    except:
        return {
                "data": image,
                "status": "Fail"
                }
