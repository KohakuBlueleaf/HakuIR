import sys 
sys.path.append("..")
import argparse
import os
import logging
from PIL import Image

logging.basicConfig(level = logging.INFO)

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
    from hakuir.image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(model_name)

    upscale_img = upscale(input_image,Image.LANCZOS)
    output = model.upscale_before_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output

def upscale_after_ir(input_image:Image,model_name:str):
    from hakuir.image_restoration import ImageRestoration

    model = ImageRestoration()
    model.load_model(model_name)

    upscale_img = upscale(input_image,Image.LANCZOS)
    output = model.upscale_after_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output

def upscale_cli(args):
    input_list = []
    output_list = []
    if os.path.isdir(args.input):
        logging.debug("Path is a directory")
        folder = args.input
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                input_list.append(path)
                output_list.append(os.path.join(args.output,file))
    else:
        input_list.append(args.input)

    logging.info("WorkList: Start")
    logging.info("WorkList: {}/{}".format(0,len(input_list)))
    for i in range(len(input_list)):
        input_img = Image.open(input_list[i]).convert('RGB')
        if args.resample == "lanczos":  
            resample = Image.LANCZOS
        logging.info("Input image loaded from {}".format(input_list[i]))
        upscale_img = upscale(input_img,resample)
        upscale_img.save(output_list[i])
        logging.info("Upscale image saved as {}".format(output_list[i]))
        logging.info("WorkList: {}/{}".format(i+1,len(input_list)))
    logging.info("WorkList: Done")

def upscale_before_ir_cli(args):
    input_list = []
    output_list = []
    if os.path.isdir(args.input):
        logging.debug("Path is a directory")
        folder = args.input
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                input_list.append(path)
                output_list.append(os.path.join(args.output,file))
    else:
        input_list.append(args.input)

    logging.info("WorkList: Start")
    logging.info("WorkList: {}/{}".format(0,len(input_list)))    
    for i in range(len(input_list)):
        logging.info("Input image loaded from {}".format(input_list[i]))
        input_img = Image.open(input_list[i]).convert('RGB')
        output = upscale_before_ir(input_img,args.model)
        output.save(output_list[i])
        logging.info("Upscale image saved as {}".format(output_list[i]))
        logging.info("WorkList: {}/{}".format(i+1,len(input_list)))
    logging.info("WorkList: Done")

def upscale_after_ir_cli(args):
    input_list = []
    output_list = []
    if os.path.isdir(args.input):
        logging.debug("Path is a directory")
        folder = args.input
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                input_list.append(path)
                output_list.append(os.path.join(args.output,file))
    else:
        input_list.append(args.input)

    logging.info("WorkList: Start")
    logging.info("WorkList: {}/{}".format(0,len(input_list)))
    for i in range(len(input_list)):
        logging.info("Input image loaded from {}".format(input_list[i]))
        input_img = Image.open(input_list[i]).convert('RGB')
        output = upscale_after_ir(input_img,args.model)
        output.save(output_list[i])
        logging.info("Upscale image saved as {}".format(output_list[i]))
        logging.info("WorkList: {}/{}".format(i+1,len(input_list)))
    logging.info("WorkList: Done")
