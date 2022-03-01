import os.path

import cv2
import numpy as np
import torch

strategies = {'jpg': cv2.IMWRITE_JPEG_QUALITY, 'jpeg': cv2.IMWRITE_JPEG_QUALITY, 'png': cv2.IMWRITE_PNG_COMPRESSION}


class CompressionLevelSet(object):
    """
    This convert changes the image compression
    """

    def __init__(self, compression_strategy='jpeg', to_compression_level=95):
        self.to_compression_level = to_compression_level
        self.compression_strategy = compression_strategy

    def __call__(self, sample):
        img, _ = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        img = _compress_img(img, self.compression_strategy, self.to_compression_level)

        sample['img'] = torch.from_numpy(img)

        return sample


def _compress_img(img, compression_strategy, to_compression_level):
    ret, encode = cv2.imencode('.' + compression_strategy, img,
                               [strategies[compression_strategy], int(to_compression_level)])
    if ret:
        return cv2.imdecode(encode, cv2.IMREAD_COLOR)
    else:
        raise Exception('Could not compress image')


if __name__ == '__main__':
    import argparse
    import glob
    import os
    import tqdm

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()
    compression = 70

    out_folder = os.path.join(opt.path, str(compression))
    os.makedirs(out_folder, exist_ok=True)

    for p in tqdm.tqdm(glob.glob(os.path.join(opt.path, '*.jpg'))):
        #print(f"Load file {p}")
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        #print(f"\tImage depth: {img.dtype}")
        #reduced_img = _compress_img(img, 'jpeg', compression)
        cv2.imwrite(os.path.join(out_folder, os.path.basename(p)), img, [int(cv2.IMWRITE_JPEG_QUALITY), compression])


