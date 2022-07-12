import argparse
import glob
import os
import cv2
import numpy
import torchvision.transforms
import tqdm

from preprocessor_factory import generate_preprocessing_pipelines, VALID_CONVERTERS, get_used_jpeg_compression_quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file or input folder")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument("converter", type=str, help=f"comma-separated list of conversions (valid conversions are: [{list(VALID_CONVERTERS.keys())}]). For more informations look into preprocessor_factory.py.")
    opt = parser.parse_args()

    converter, _, _ = generate_preprocessing_pipelines(opt.converter, only_image_input=True)
    used_jpeg_compression = get_used_jpeg_compression_quality(converter)
    converter = torchvision.transforms.Compose(converter)
    os.makedirs(opt.output, exist_ok=True)
    if '.png' in opt.input or '.jpg' in opt.input:
        img_paths = glob.glob(os.path.join(opt.input))
    else:
        img_paths = glob.glob(os.path.join(opt.input, '*'))

    for img_path in tqdm.tqdm(img_paths):
        img = cv2.imread(img_path)
        img = converter(img)
        if not isinstance(img, numpy.ndarray):
            img = img.numpy()
        if used_jpeg_compression:
            # apply the jpeg compression also during the storing procedure 
            img_path = img_path.replace(".png", ".jpg")
            cv2.imwrite(os.path.join(opt.output, os.path.basename(img_path)), img, [int(cv2.IMWRITE_JPEG_QUALITY), used_jpeg_compression])
        else:
            cv2.imwrite(os.path.join(opt.output, os.path.basename(img_path)), img)
