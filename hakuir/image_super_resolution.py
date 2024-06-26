import math
import toml

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
from PIL import Image

from .utils import instantiate
from .utils.img_process import match_color_pil


class ImageSuperRes:
    def __init__(self) -> None:
        self.model = None
        self.scale = 2
        self.tile_size = 512
        self.tile_overlap = 16

    def load_model(self, model_name="RGT_x2"):
        config = toml.load(f"./models/{model_name}.toml")["network"]
        state_dict = torch.load(f"./models/{model_name}.pth")
        if "params" in state_dict:
            state_dict = state_dict["params"]

        net: nn.Module = instantiate(f"models.{config['class']}")(**config["configs"])
        net.load_state_dict(state_dict)
        net = net.cuda()
        self.model = net
        self.scale = config.get("scale", self.scale)
        self.tile_size = config.get("img_size", self.tile_size)

    @torch.no_grad()
    def _restoration(self, img: torch.Tensor, batch_size=4):
        # test the image tile by tile
        b, c, h, w = img.size()
        tile = min(self.tile_size, h, w)

        stride = tile - self.tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(
            b, c, h * self.scale, w * self.scale, dtype=img.dtype, device=img.device
        ).type_as(img)
        W = torch.zeros_like(E, dtype=img.dtype, device=img.device)

        all_patch = []
        all_idx = []

        with tqdm(
            total=math.ceil(len(h_idx_list) * len(w_idx_list) / batch_size),
            desc="SR tiles",
            leave=False,
        ) as pbar:
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                    all_patch.append(in_patch)
                    all_idx.append((h_idx, w_idx))
            for i in range(0, len(all_patch), batch_size):
                current_patch = torch.cat(all_patch[i : i + batch_size])
                out_patch = self.model(current_patch)
                for idx, (h_idx, w_idx) in enumerate(all_idx[i : i + batch_size]):
                    all_patch[i + idx] = (
                        h_idx * self.scale,
                        w_idx * self.scale,
                        out_patch[idx],
                    )
                pbar.update(1)
                mem_usage = torch.cuda.memory_allocated() / 1e6
                pbar.set_postfix({"mem": f"{mem_usage:.1f}MB"})

        tile = int(tile * self.scale)
        for h_idx, w_idx, out_patch in tqdm(
            all_patch,
            desc="Rebuild tiles",
            leave=False,
        ):
            out_patch_mask = torch.ones_like(out_patch)
            E[..., h_idx : (h_idx + tile), w_idx : (w_idx + tile)].add_(out_patch)
            W[..., h_idx : (h_idx + tile), w_idx : (w_idx + tile)].add_(out_patch_mask)

        output = E.div_(W)
        return output

    @torch.no_grad()
    def restoration(
        self, img: Image.Image, batch_size=4, device="cuda", dtype=torch.float16
    ) -> Image.Image:
        img_tensor = ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0).cuda()
        with torch.autocast(device, dtype):
            output: torch.Tensor = self._restoration(img_tensor, batch_size)
        output = output.squeeze(0).permute(1, 2, 0).float().cpu()
        output = torch.clamp(output, 0, 1)
        output = (output * 255).numpy().astype(np.uint8)
        output: Image.Image = Image.fromarray(output)
        output = match_color_pil(output, img)
        return output

    def upscale(
        self,
        img: Image.Image,
        scale=2,
        batch_size=4,
        device="cuda",
        dtype=torch.float16,
    ) -> Image.Image:
        target_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        upscale = img
        for _ in trange(
            math.ceil(math.log(scale, self.scale)), desc="Upscale Pass", leave=False
        ):
            upscale = self.restoration(upscale, batch_size, device, dtype)

        upscale = upscale.resize(target_size, resample=Image.BICUBIC)
        return upscale
