import cv2
import torch


class Downscale(object):
    """
    This convert changes the image compression
    """

    def __init__(self, max_side_length):
        self.max_side_length = int(max_side_length)

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        width, height, _ = img.shape

        if height > width:
            scale = self.max_side_length / height
            resized_height = self.max_side_length
            resized_width = int(width * scale)
        else:
            scale = self.max_side_length / width
            resized_height = int(height * scale)
            resized_width = self.max_side_length

        img = cv2.resize(img, (resized_height, resized_width))

        if len(annot) > 0:
            annot[:, 0] *= scale
            annot[:, 1] *= scale
            annot[:, 2] *= scale
            annot[:, 3] *= scale

        sample['img'] = torch.from_numpy(img)
        sample['annot'] = annot

        return sample
