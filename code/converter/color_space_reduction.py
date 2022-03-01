import cv2
import numpy as np
import torch


class ColorSpaceReduction(object):
    """
    This convert reduces the bit depth of the recordings
    """
    def __init__(self, output_type):
        self.output_type = output_type

    def __call__(self, sample):
        img, _ = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if self.output_type == 'rgb':
            img = _convert_bgrne_to_rgb(img)

        sample['img'] = torch.from_numpy(img)

        return sample


def _convert_bgrne_to_rgb(img):
    """
        Converts a multispectral record to a rgb image
        ! CV visualize BGR by default
    """
    assert len(img.shape) == 3
    assert img.shape[-1] == 5

    reduced_img = img[:, :, (2, 1, 0)]

    # normalize between 0-255
    reduced_img = ((reduced_img - reduced_img.min()) / (reduced_img.max() - reduced_img.min()) * 255).astype(np.uint8)

    return reduced_img


if __name__ == '__main__':
    import argparse
    from skimage import io

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = io.imread(opt.path)
    reduced_img = _convert_bgrne_to_rgb(img)

    # see comment
    reduced_img = cv2.cvtColor(reduced_img, cv2.COLOR_BGR2RGB)

    cv2.imshow("ColorSpace recduced", reduced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
