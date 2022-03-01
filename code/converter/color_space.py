import cv2
import numpy as np
import torch

color_spaces = ['RGB', 'BGR', 'GRAY','HLS','HSV','XYZ','LUV','YUV','YCrCb']

class ColorSpaceConverter(object):
    """
    This convert convers the color space
    """
    def __init__(self, to_colorspace, from_colorspace='RGB'):
        self.to_colorspace = to_colorspace
        self.from_colorspace = from_colorspace
        if to_colorspace not in color_spaces or from_colorspace not in color_spaces:
            raise NotImplementedError

    def __call__(self, sample):
        img, _  = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        img = _convert_color_space(img, self.from_colorspace, self.to_colorspace)

        sample['img'] = torch.from_numpy(img)

        return sample


def _convert_color_space(img, from_colorspace, to_colorspace):
    if from_colorspace == to_colorspace:
         return img
    colorspace_conversion_arg = 'cv2.COLOR_'+from_colorspace+'2'+to_colorspace
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img.astype(np.uint8),eval(colorspace_conversion_arg))
    if to_colorspace == 'GRAY':
         img = np.stack((img,)*3, axis=-1)
    return img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    from_color = 'RGB'
    to_color = 'GRAY'
    reduced_img = _convert_color_space(img, from_color, to_color)
    concat = np.concatenate((img, reduced_img), axis=1)

    print(f"Convert the input image from {from_color} to {to_color}")
    cv2.imshow("ColorSpace compare", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
