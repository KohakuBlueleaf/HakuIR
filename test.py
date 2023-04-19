import env
from PIL import Image
from image_restoration import ImageRestoration


if __name__ == '__main__':
    model = ImageRestoration()
    model.load_model('NAFNet-REDS-width64')
    
    test_img = Image.open('./demo/original.png').convert('RGB')
    test_upscale = test_img.resize(
        (test_img.size[0]*2,test_img.size[1]*2),
        resample=Image.LANCZOS
    )
    test_upscale.save('./demo/lanczos.png')
    
    output = model.upscale_after_ir(test_img)
    output = Image.blend(test_upscale, output, 0.5)
    output.save('./demo/ir-begin.png')
    
    output = model.upscale_before_ir(test_img)
    output = Image.blend(test_upscale, output, 0.5)
    output.save('./demo/ir-end.png')