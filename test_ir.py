import hakuir.env
from PIL import Image
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from hakuir.image_restoration import ImageRestoration

SCALE = 1


if __name__ == "__main__":
    model = ImageRestoration()
    model.tile_size = 256
    model.tile_overlap = 32
    model.load_model("NAFNet-REDS-width64")

    model2 = ImageRestoration()
    model2.tile_size = 256
    model2.tile_overlap = 32
    model2.load_model("SCUNet-PSNR")

    test_img = Image.open("./demo/gnoise.png").convert("RGB")
    test_img = to_tensor(test_img)
    test_img = test_img + torch.randn_like(test_img) * 0.02
    test_img = torch.clamp(test_img, 0, 1)
    test_img = (test_img.transpose(0, 2).transpose(0, 1) * 255).numpy().astype(np.uint8)
    test_img = Image.fromarray(test_img)
    test_upscale = test_img.resize(
        (int(test_img.size[0] * SCALE), int(test_img.size[1] * SCALE)),
        resample=Image.LANCZOS,
    )
    test_upscale.save("./demo/lanczos.png")

    output2 = test_img
    output2 = model2.upscale_after_ir(test_img, 1, 16)
    output2.save("./demo/scunet-begin.png")

    output = model.upscale_after_ir(output2, SCALE, 8)
    output.save("./demo/ir-begin.png")

    output = model.upscale_before_ir(output2, SCALE, 8)
    output.save("./demo/ir-end.png")
