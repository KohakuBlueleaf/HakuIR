import argparse
from .commands import available_model_list, download_models, \
    upscale_console, restoration_console, \
    upscale_before_ir_console, upscale_after_ir_console


def console():
    parser = argparse.ArgumentParser(description='HAKUIR')

    subparsers = parser.add_subparsers(metavar='Command')
    models_command = subparsers.add_parser(
        'models', help='Output model list')
    models_command.add_argument(
        '-a', '--available', action='store_true',
        help='Output available model list')
    models_command.set_defaults(handle=available_model_list)

    download_command = subparsers.add_parser(
        'download', help='Download model')
    download_command.add_argument(
        '-m', '--model', type=str, required=True,
        help='Model name')
    download_command.set_defaults(handle=download_models)

    upscale_command = subparsers.add_parser(
        'upscale', help='Just upscale')
    upscale_command.add_argument(
        '-r', '--resample', type=str, required=True,
        help='resample method name')
    upscale_command.add_argument(
        '-i', '--input', type=str, required=True,
        help='input image path')
    upscale_command.add_argument(
        '-o', '--output', type=str,
        help='output image path')
    upscale_command.set_defaults(handle=upscale_console)

    restoration_command = subparsers.add_parser(
        'restoration', help='Just restoration')
    restoration_command.add_argument(
        '-m', '--model', default='NAFNet-REDS-width64', type=str,
        help='Restoration Model. Use hakuir models to see available models')
    restoration_command.add_argument(
        '-i', '--input', type=str, required=True,
        help='input image path')
    restoration_command.add_argument(
        '-o', '--output', type=str,
        help='output image path')
    restoration_command.set_defaults(handle=restoration_console)

    upscale_before_ir_command = subparsers.add_parser(
        'upscale-before-ir', help='Upscale before restoration')
    upscale_before_ir_command.add_argument(
        '-m', '--model', default='NAFNet-REDS-width64', type=str,
        help='Restoration Model. Use hakuir models to see available models')
    upscale_before_ir_command.add_argument(
        '-i', '--input', type=str, required=True,
        help='input image path')
    upscale_before_ir_command.add_argument(
        '-o', '--output', type=str,
        help='output image path')
    upscale_before_ir_command.set_defaults(handle=upscale_before_ir_console)

    upscale_after_ir_command = subparsers.add_parser(
        'upscale-after-ir', help='Upscale after restoration')
    upscale_after_ir_command.add_argument(
        '-m', '--model', default='NAFNet-REDS-width64', type=str,
        help='Restoration Model. Use hakuir models to see available models')
    upscale_after_ir_command.add_argument(
        '-i', '--input', type=str, required=True,
        help='input image path')
    upscale_after_ir_command.add_argument(
        '-o', '--output', type=str,
        help='output image path')
    upscale_after_ir_command.set_defaults(handle=upscale_after_ir_console)

    args = parser.parse_args()

    if hasattr(args, 'handle'):
        args.handle(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    console()
