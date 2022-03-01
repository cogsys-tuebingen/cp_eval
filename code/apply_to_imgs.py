import argparse
import glob
import os
import cv2
import numpy
import torchvision.transforms
import tqdm

from preprocessor_factory import generate_preprocessing_pipelines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("converter", type=str)

    opt = parser.parse_args()

    converter, _, _ = generate_preprocessing_pipelines(opt.converter, only_image_input=True)
    converter = torchvision.transforms.Compose(converter)
    os.makedirs(opt.output, exist_ok=True)
    img_paths = glob.glob(os.path.join(opt.input, '*.jpg')) + glob.glob(os.path.join(opt.input, '*.png'))

    for img_path in tqdm.tqdm(img_paths):
        img = cv2.imread(img_path)
        img = converter(img)
        if not isinstance(img, numpy.ndarray):
            img = img.numpy()
        img_path = img_path.replace(".png", ".jpg")
        cv2.imwrite(os.path.join(opt.output, os.path.basename(img_path)), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#        cv2.imwrite(os.path.join(opt.output, os.path.basename(img_path)), img)
