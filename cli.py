import argparse
import os
from commands import available_model_list, upscale, upscale_before_ir, upscale_after_ir

def cli():
    parser = argparse.ArgumentParser(description="HAKUIR")

    subparsers = parser.add_subparsers(metavar="Command")
    modelsCommand = subparsers.add_parser("models", help="Output model list")
    modelsCommand.set_defaults(handle=available_model_list)

    upscaleCommand = subparsers.add_parser("upscale", help="Just upscale")
    upscaleCommand.add_argument("-r","--resample",type=str,required=True, help="resample method name")
    upscaleCommand.add_argument("-i","--input",type=str,required=True, help="input image path")
    upscaleCommand.add_argument("-o","--output",type=str,required=True, help="output image path")
    upscaleCommand.set_defaults(handle=upscale)

    upscaleBeforeIRCommand = subparsers.add_parser("upscale-before-ir", help="Upscale before IR")
    upscaleBeforeIRCommand.add_argument("-m","--model",type=str,help="Restoration Model. Use models command to see available models")
    upscaleBeforeIRCommand.add_argument("-i","--input",type=str,required=True, help="input image path")
    upscaleBeforeIRCommand.add_argument("-o","--output",type=str,required=True, help="output image path")
    upscaleBeforeIRCommand.set_defaults(handle=upscale_before_ir)

    upscaleAfterIRCommand = subparsers.add_parser("upscale-after-ir", help="Upscale after IR")
    upscaleAfterIRCommand.add_argument("-m","--model",type=str,help="Restoration Model. Use models command to see available models")
    upscaleAfterIRCommand.add_argument("-i","--input",type=str,required=True, help="input image path")
    upscaleAfterIRCommand.add_argument("-o","--output",type=str,required=True, help="output image path")
    upscaleAfterIRCommand.set_defaults(handle=upscale_after_ir)

    args = parser.parse_args()

    if hasattr(args, "handle"):
        args.handle(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()