import cv2
import numpy as np
from PIL import Image


def match_color_pil(source, target):
    target = target.resize(source.size)
    source = source.convert("RGB")
    target = target.convert("RGB")
    source = np.array(source)[:, :, ::-1].copy()
    target = np.array(target)[:, :, ::-1].copy()
    result = match_color(source, target)[:, :, ::-1]
    return Image.fromarray(result)


def match_color(source, target):
    # Convert RGB to L*a*b*, and then match the std/mean
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    source = cv2.cvtColor(
        (result * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    source = source.astype(np.float32)
    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0])
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1])
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2])
    output = source
    return output.clip(0, 255).astype(np.uint8)


def wavelet_colorfix(inp, target):
    inp_high, _ = wavelet_decomposition(inp, 5)
    _, target_low = wavelet_decomposition(target, 5)
    output = inp_high + target_low
    return output


def wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def wavelet_blur(inp, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(inp, (kernel_size, kernel_size), 0)
    return output
