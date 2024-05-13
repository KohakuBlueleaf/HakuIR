import hakuir.env

import torch
import numpy as np

from PIL import Image
from torchvision.transforms.functional import to_tensor

from hakuir.image_super_resolution import ImageSuperRes

SCALE = 4


if __name__ == "__main__":
    test_img = Image.open("./demo/lowres.png").convert("RGB")
    test_img = to_tensor(test_img)
    test_img = torch.clamp(test_img, 0, 1)
    test_img = (test_img.transpose(0, 2).transpose(0, 1) * 255).numpy().astype(np.uint8)
    test_img = Image.fromarray(test_img)
    test_upscale = test_img.resize(
        (int(test_img.size[0] * SCALE), int(test_img.size[1] * SCALE)),
        resample=Image.LANCZOS,
    )
    test_upscale.save("./demo/lanczos-upscale-4x.png")

    for i in range(2, 5):
        model = ImageSuperRes()
        model.load_model(f"RGT_x{i}")
        output = model.upscale(test_img, SCALE, 8, dtype=torch.float32)
        output.save(f"./demo/RGT_x{i}-upscale-4x.png")
