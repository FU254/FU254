"""Published MGCA and BioViL-T evaluation geometry after shared DICOM display."""

import numpy as np
from PIL import Image


def resize_extra(pixels, preprocess):
    get = (lambda key, default=None: preprocess.get(key, default)) if isinstance(preprocess, dict) else (
        lambda key, default=None: getattr(preprocess, key, default))
    mode, size = get("resize_mode"), int(get("image_size"))
    if get("interpolation", "bicubic") != "bilinear" or get("do_hist_equalization", False):
        raise ValueError("Extra VLM geometry must retain its reviewed interpolation and no histogram equalization")
    if mode == "mgca_pad256_center_crop":
        if size != 224:
            raise ValueError("MGCA's reviewed pad256/crop224 geometry requires 224 input")
        import cv2

        image = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        height, width = image.shape
        if height >= width:
            new_height, new_width = 256, max(1, int(width * (256.0 / height)))
        else:
            new_height, new_width = max(1, int(height * (256.0 / width))), 256
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        top, left = (256 - new_height) // 2, (256 - new_width) // 2
        canvas = np.pad(resized, ((top, 256-new_height-top), (left, 256-new_width-left)))
        return canvas[16:240, 16:240].astype(np.float32) / 255.0
    if mode == "biovil_center_crop":
        if size != 448 or get("resize_shortest_edge") != 512:
            raise ValueError("BioViL-T requires short edge 512, center crop 448")
        # torchvision's published Resize(int) + CenterCrop uses round(),
        # unlike the existing framework mode's floor-based center crop.
        image = Image.fromarray(np.asarray(pixels, dtype=np.float32))
        width, height = image.size
        dimensions = ((512, int(512 * height / width)) if width <= height
                      else (int(512 * width / height), 512))
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
        left, top = int(round((image.width-size)/2)), int(round((image.height-size)/2))
        return np.clip(np.asarray(image.crop((left, top, left+size, top+size)), dtype=np.float32), 0, 1)
    raise ValueError("Unknown extra VLM resize mode")
