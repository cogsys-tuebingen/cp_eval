import cv2
import numpy as np
import torch


class DynamicGammaConverter(object):
    """
    This convert adjusts the gamma of the recordings
    """
    def __init__(self):
        pass
    
    def __call__(self, sample):
        img, _  = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        img = _adjust_gamma_dynamic(img.astype(np.uint8))

        sample['img'] = torch.from_numpy(img)

        return sample

def _heaviside(x):
    return 1 if x > 0 else 0


def _adjust_gamma_dynamic(img):
    """https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0138-1#CR14 
	Applies adaptive Gamma-Correction

    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    I_in = v/255.0
    I_out = I_in

    sigma = np.std(I_in)
    mean = np.mean(I_in)
    D = 4*sigma

    if D <= 1/3:
        gamma = - np.log2(sigma)

        I_in_f = I_in**gamma
        mean_f = (mean**gamma)

        k =  I_in_f + (1 - I_in_f) * mean_f
        c = 1 / (1 + _heaviside(0.5 - mean) * (k-1))

        if mean < 0.5:
            I_out = I_in_f / ((I_in_f + ((1-I_in_f) * mean_f)))
        else:
            I_out = c * I_in_f

    elif D > 1/3:
        gamma = np.exp((1- (mean+sigma))/2)

        I_in_f = I_in**gamma
        mean_f = (mean**gamma)

        k =  I_in_f + (1 - I_in_f) * mean_f
        c = 1/ (1 + _heaviside(0.5 - mean) * (k-1))
        I_out = c * I_in_f

    I_out = I_out*255

    new_v = I_out.astype(np.uint8)

    gamma_corrected_v = cv2.merge((h, s, new_v))
    gamma_corrected = cv2.cvtColor(gamma_corrected_v, cv2.COLOR_HSV2RGB)
    return gamma_corrected



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
    adjusted_img = _adjust_gamma_dynamic(img)


    concat = np.concatenate((img, adjusted_img), axis=1)

    cv2.imshow("Gamma", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
