import argparse
import os
import logging
from PIL import Image

def available_model_list(args):
    folder = './models'
    model_list = []
    config_list= []
    available_models = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            if path.endswith(".txt"):
                continue
            elif path.endswith(".pth"):
                model_list.append(os.path.splitext(file)[0])
            elif path.endswith(".toml"):
                config_list.append(os.path.splitext(file)[0])

    for model in config_list:
        if model in model_list:
            available_models.append(model)
    print(available_models)
    
def upscale(args):
    if os.path.isdir(args.input):
        logging.ERROR("Path is a directory")
        return

    input_img = Image.open(args.input).convert('RGB')

    if args.resample == "lanczos":  
        resample = Image.LANCZOS

    logging.info("Input image loaded from {}".format(args.input))
    upscale_img = input_img.resize(
        (input_img.size[0]*2,input_img.size[1]*2),
        resample
    )
    upscale_img.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))

def upscale_before_ir(args):
    if os.path.isdir(args.input):
        logging.ERROR("Path is a directory")
        return
    
    from image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(args.model)

    logging.info("Input image loaded from {}".format(args.input))
    input_img = Image.open(args.input).convert('RGB')
    upscale_img = input_img.resize(
        (input_img.size[0]*2,input_img.size[1]*2),
        resample=Image.LANCZOS
    )

    output = model.upscale_before_ir(input_img)
    output = Image.blend(upscale_img, output, 0.5)
    output.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))

def upscale_after_ir(args):
    if os.path.isdir(args.input):
        logging.ERROR("Path is a directory")
        return
    
    from image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(args.model)
    
    logging.info("Input image loaded from {}".format(args.input))
    input_img = Image.open(args.input).convert('RGB')
    upscale_img = input_img.resize(
        (input_img.size[0]*2,input_img.size[1]*2),
        resample=Image.LANCZOS
    )

    output = model.upscale_after_ir(input_img)
    output = Image.blend(upscale_img, output, 0.5)
    output.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))
