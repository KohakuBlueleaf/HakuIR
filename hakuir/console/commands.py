import sys
import os
import logging
from PIL import Image
import urllib.request
import time

sys.path.append('..')

logging.basicConfig(level=logging.INFO)

downloadable_model_list = {
    'SCUNet-GAN': 'scunet_color_real_gan',
    'SCUNet-PSNR': 'scunet_color_real_psnr',
}


def available_model_list(args):
    if args.available:
        logging.info('Available models:')
        for model in downloadable_model_list:
            logging.info(model)
        return downloadable_model_list
    else:
        return local_model_list(args)


def local_model_list(args):
    folder = './models'
    model_list = []
    config_list = []
    available_models = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            if path.endswith('.txt'):
                continue
            elif path.endswith('.pth'):
                model_list.append(os.path.splitext(file)[0])
            elif path.endswith('.toml'):
                config_list.append(os.path.splitext(file)[0])

    for model in config_list:
        if model in model_list:
            available_models.append(model)
    logging.info(available_models)
    return available_models


def download_models(args):
    model_dir = './models'
    model_name = args.model + '.pth'
    if os.path.exists(os.path.join(model_dir, model_name)):
        logging.info('Model {} already exists'.format(model_name))
        return
    os.makedirs(model_dir, exist_ok=True)
    url = 'https://github.com/cszn/KAIR/releases/download/v1.0/{}'.format(
        downloadable_model_list[args.model] + '.pth'
    )
    logging.info('Downloading {}...'.format(model_name))
    urllib.request.urlretrieve(url, os.path.join(model_dir, model_name))
    logging.info('Downloaded!')


def upscale(input_image: Image, resample: Image.Resampling):
    upscale_img = input_image.resize(
        (input_image.size[0] * 2, input_image.size[1] * 2), resample
    )
    return upscale_img


def restoration(input_image: Image, ir):
    output = ir.restoration(input_image)
    output = Image.blend(output, output, 0.5)
    return output


def upscale_before_ir(input_image: Image, ir):
    upscale_img = upscale(input_image, Image.LANCZOS)
    output = ir.upscale_before_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output


def upscale_after_ir(input_image: Image, ir):
    upscale_img = upscale(input_image, Image.LANCZOS)
    output = ir.upscale_after_ir(input_image)
    output = Image.blend(upscale_img, output, 0.5)
    return output


def check_path(input, output):
    input_list = []
    output_list = []

    if os.path.isdir(input):
        logging.debug('Path is a directory')
        folder = input
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                input_list.append(path)
                if output == '' or output is None:
                    logging.debug('Output is missed,use default')
                    default_out = os.path.splitext(path)[0] + '_upscaled.png'
                    output_list.append(default_out)
                elif os.path.isdir(output):
                    output_list.append(os.path.join(output, file))
    else:
        input_list.append(input)
        if output == '' or output is None:
            logging.debug('Output is missed,use default')
            output_list.append(os.path.splitext(input)[0] + '_upscaled.png')
        elif os.path.isdir(output):
            output_list.append(os.path.join(output, os.path.basename(input)))
    return input_list, output_list


def upscale_console(args):
    input_list, output_list = check_path(args.input, args.output)

    logging.info('WorkList: Start')
    time_start = time.time()
    logging.info('WorkList: {}/{}'.format(0, len(input_list)))
    for i in range(len(input_list)):
        input_img = Image.open(input_list[i]).convert('RGB')
        if args.resample == 'lanczos':
            resample = Image.LANCZOS
        logging.info('Input image loaded from {}'.format(input_list[i]))
        upscale_img = upscale(input_img, resample)
        upscale_img.save(output_list[i])
        logging.info('Upscale image saved as {}'.format(output_list[i]))
        logging.info('WorkList: {}/{}'.format(i + 1, len(input_list)))
    time_end = time.time()
    time_cost = time_end - time_start
    logging.info('WorkList: Done in {}s'.format(time_cost))


def restoration_console(args):
    input_list, output_list = check_path(args.input, args.output)

    logging.info('Prepare: Load Restoration Model')
    from hakuir.image_restoration import ImageRestoration

    ir = ImageRestoration()
    ir.load_model(args.model)
    logging.info('Prepare: Done')

    logging.info('WorkList: Start')
    time_start = time.time()
    logging.info('WorkList: {}/{}'.format(0, len(input_list)))
    for i in range(len(input_list)):
        logging.info('Input image loaded from {}'.format(input_list[i]))
        input_img = Image.open(input_list[i]).convert('RGB')
        output = restoration(input_img, ir)
        output.save(output_list[i])
        logging.info('Upscale image saved as {}'.format(output_list[i]))
        logging.info('WorkList: {}/{}'.format(i + 1, len(input_list)))
    time_end = time.time()
    time_cost = time_end - time_start
    logging.info('WorkList: Done in {}s'.format(time_cost))


def upscale_before_ir_console(args):
    input_list, output_list = check_path(args.input, args.output)

    logging.info('Prepare: Load Restoration Model')
    from hakuir.image_restoration import ImageRestoration

    ir = ImageRestoration()
    ir.load_model(args.model)
    logging.info('Prepare: Done')

    logging.info('WorkList: Start')
    time_start = time.time()
    logging.info('WorkList: {}/{}'.format(0, len(input_list)))
    for i in range(len(input_list)):
        logging.info('Input image loaded from {}'.format(input_list[i]))
        input_img = Image.open(input_list[i]).convert('RGB')
        output = upscale_before_ir(input_img, ir)
        output.save(output_list[i])
        logging.info('Upscale image saved as {}'.format(output_list[i]))
        logging.info('WorkList: {}/{}'.format(i + 1, len(input_list)))
    time_end = time.time()
    time_cost = time_end - time_start
    logging.info('WorkList: Done in {}s'.format(time_cost))


def upscale_after_ir_console(args):
    input_list, output_list = check_path(args.input, args.output)

    logging.info('Prepare: Load Restoration Model')
    from hakuir.image_restoration import ImageRestoration

    ir = ImageRestoration()
    ir.load_model(args.model)
    logging.info('Prepare: Done')

    logging.info('WorkList: Start')
    time_start = time.time()
    logging.info('WorkList: {}/{}'.format(0, len(input_list)))
    for i in range(len(input_list)):
        logging.info('Input image loaded from {}'.format(input_list[i]))
        input_img = Image.open(input_list[i]).convert('RGB')
        output = upscale_after_ir(input_img, ir)
        output.save(output_list[i])
        logging.info('Upscale image saved as {}'.format(output_list[i]))
        logging.info('WorkList: {}/{}'.format(i + 1, len(input_list)))
    time_end = time.time()
    time_cost = time_end - time_start
    logging.info('WorkList: Done in {}s'.format(time_cost))
