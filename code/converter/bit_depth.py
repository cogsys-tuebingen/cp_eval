import cv2
import numpy as np
import torch


class BitDepthConverter(object):
    """
    This convert reduces the bit depth of the recordings
    """
    def __init__(self, to_bit_depth, from_bit_depth=8):
        self.to_bit_depth = to_bit_depth
        self.from_bit_depth = from_bit_depth

    def __call__(self, sample):
        img, _  = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        img = _reduce_bit_depth(img, self.from_bit_depth, self.to_bit_depth,
                utilize_full_dtype=True)

        sample['img'] = torch.from_numpy(img)

        return sample


def _get_max_value(bit_depth):
    return pow(2, bit_depth) - 1


def _reduce_bit_depth(img, from_bit_depth, to_bit_depth, utilize_full_dtype=False):
    org_dtype = img.dtype
    conversion_factor = float(_get_max_value(to_bit_depth)) / float(_get_max_value(from_bit_depth))
    img = img.copy() * conversion_factor
    img = img.astype(org_dtype)

    if utilize_full_dtype:
        img = img / conversion_factor
        img = img.astype(org_dtype)

    return img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    print(f"\tImage depth: {img.dtype}")
    reduced_img = _reduce_bit_depth(img, 8, 2, utilize_full_dtype=True)
    concat = np.concatenate((img, reduced_img), axis=1)

    print("Convert the input image to a 2 bit depth")
    cv2.imshow("2 Bit depth image", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
