import cv2
import numpy as np
import torch


class GammaConverter(object):
    """
    This convert adjusts the gamma of the recordings
    """

    def __init__(self, to_gamma):
        self.to_gamma = to_gamma

    def __call__(self, sample):
        img, _ = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        img = _adjust_gamma(img.astype(np.uint8), self.to_gamma)

        sample['img'] = torch.from_numpy(img)

        return sample


def _adjust_gamma(img, to_gamma):
    invGamma = 1.0 / to_gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    adjusted_img = _adjust_gamma(img, 0.5)
    concat = np.concatenate((img, adjusted_img), axis=1)

    cv2.imshow("Gamma", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
