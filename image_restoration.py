import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

import toml
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

from utils import instantiate


class ImageRestoration:
    def __init__(self) -> None:
        self.model = None
        self.tile_size = 512
        self.tile_overlap = 16
    
    def load_model(self, model_name='NAFNet-REDS-width64'):
        config = toml.load(f'./models/{model_name}.toml')['network']
        state_dict = torch.load(f'./models/{model_name}.pth')
        if 'params' in state_dict:
            state_dict = state_dict['params']
        
        net: nn.Module = instantiate(config['class'])(**config['configs'])
        net.load_state_dict(state_dict)
        net = net.cuda().half()
        self.model = net
    
    @torch.no_grad()
    def restoration(self, img: torch.Tensor):
        # test the image tile by tile
        b, c, h, w = img.size()
        tile = min(self.tile_size, h, w)
        
        stride = tile - self.tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w, dtype=img.dtype, device=img.device).type_as(img)
        W = torch.zeros_like(E, dtype=img.dtype, device=img.device)
        
        with tqdm(total=len(h_idx_list) * len(w_idx_list), desc="IR tiles") as pbar:
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[
                    ..., h_idx: (h_idx + tile), w_idx: (w_idx + tile)
                    ].add_(out_patch)
                    W[
                    ..., h_idx: (h_idx + tile), w_idx: (w_idx + tile)
                    ].add_(out_patch_mask)
                    pbar.update(1)
                    mem_usage = torch.cuda.memory_allocated()/1e6
                    pbar.set_postfix({'mem': f'{mem_usage:.1f}MB'})
        output = E.div_(W)
        return output
    
    @torch.no_grad()
    def upscale_before_ir(
        self,
        img: Image.Image, 
        scale = 2, 
        device = 'cuda', 
        dtype = torch.float16
    ) -> Image.Image:
        target_size = (
            int(img.size[0]*scale),
            int(img.size[1]*scale)
        )
        
        upscale = img.resize(
            target_size,
            resample=Image.LANCZOS
        )
        upscale = ToTensor()(upscale)
        upscale = upscale.unsqueeze(0).to(device=device, dtype=dtype)
        
        with torch.autocast(device, dtype):
            output: torch.Tensor = self.restoration(upscale)
        output = output.squeeze(0).permute(1, 2, 0).float().cpu()
        output = torch.clamp(output, 0, 1)
        output = (output*255).numpy().astype(np.uint8)
        output: Image.Image = Image.fromarray(output)
        
        return output
    
    @torch.no_grad()
    def upscale_after_ir(
        self,
        img: Image.Image, 
        scale = 2, 
        device = 'cuda', 
        dtype = torch.float16
    ) -> Image.Image:
        target_size = (
            int(img.size[0]*scale),
            int(img.size[1]*scale)
        )
        img = ToTensor()(img)
        img = img.unsqueeze(0).to(device=device, dtype=dtype)
        
        with torch.autocast(device, dtype):
            output: torch.Tensor = self.restoration(img)
        output = output.squeeze(0).permute(1, 2, 0).float().cpu()
        output = torch.clamp(output, 0, 1)
        output = (output*255).numpy().astype(np.uint8)
        output: Image.Image = Image.fromarray(output)
        
        output = output.resize(
            target_size,
            resample=Image.LANCZOS
        )
        return output