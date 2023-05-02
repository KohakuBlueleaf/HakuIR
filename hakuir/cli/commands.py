import sys
import os
import logging
from PIL import Image
import time

sys.path.append('..')

logging.basicConfig(level=logging.INFO)


def available_model_list(args):
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
    print(available_models)


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
                    default_out = os.path.splitext(path)[0] + '_upscaled.png'
                    output_list.append(default_out)
                elif os.path.isdir(output):
                    output_list.append(os.path.join(output, file))
    else:
        input_list.append(input)
        if output == '' or output is None:
            output_list.append(os.path.splitext(input)[0] + '_upscaled.png')
        elif os.path.isdir(output):
            output_list.append(os.path.join(output, os.path.basename(input)))
    return input_list, output_list


def upscale_cli(args):
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


def restoration_cli(args):
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


def upscale_before_ir_cli(args):
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


def upscale_after_ir_cli(args):
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
