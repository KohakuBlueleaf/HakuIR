import sys 
sys.path.append("..")
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

def upscale(input_image:Image,resample:Image.Resampling):
    upscale_img = input_image.resize((input_image.size[0]*2,input_image.size[1]*2),resample)
    return upscale_img

def upscale_before_ir(input_image:Image,model_name:str):
    from image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(model_name)

    upscale_img = upscale(input_image,Image.LANCZOS)
    output = model.upscale_before_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output

def upscale_after_ir(input_image:Image,model_name:str):
    from image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(model_name)

    upscale_img = upscale(input_image,Image.LANCZOS)
    output = model.upscale_after_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output

def upscale_cli(args):
    if os.path.isdir(args.input):  
        logging.ERROR("Path is a directory")
        return

    input_img = Image.open(args.input).convert('RGB')

    if args.resample == "lanczos":  
        resample = Image.LANCZOS

    logging.info("Input image loaded from {}".format(args.input))
    upscale_img = upscale(input_img,resample)
    upscale_img.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))

def upscale_before_ir_cli(args):
    if os.path.isdir(args.input):
        logging.ERROR("Path is a directory")
        return
    
    logging.info("Input image loaded from {}".format(args.input))
    input_img = Image.open(args.input).convert('RGB')
    output = upscale_before_ir(input_img,args.model)
    output.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))

def upscale_after_ir_cli(args):
    if os.path.isdir(args.input):
        logging.ERROR("Path is a directory")
        return
    logging.info("Input image loaded from {}".format(args.input))
    input_img = Image.open(args.input).convert('RGB')
    output = upscale_after_ir(input_img,args.model)
    output.save(args.output)
    logging.info("Upscale image saved as {}".format(args.output))
