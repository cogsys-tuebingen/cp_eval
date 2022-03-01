import cv2
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
from matplotlib import pyplot as plt

class LensDistortionSim(object):
    """
    This convert convers the color space
    """
    def __init__(self, distCoeff):
        self.distCoeff = distCoeff
       
    def __call__(self, sample):
        img, annot  = sample['img'], sample['annot']

        if isinstance(img, torch.Tensor):
            img = img.numpy()
            annot = annot.numpy()

        img, new_annots = _sim_distortion(img, annot, self.distCoeff)

        sample['img'] = torch.from_numpy(img)
        sample['annot'] = torch.from_numpy(new_annots)

        return sample


def _bounding_box(points,img):
    x_coordinates, y_coordinates = zip(*points)
    x_coordinates = np.clip(x_coordinates, 0, img.shape[1])
    y_coordinates = np.clip(y_coordinates, 0, img.shape[0])
    x1, y1, x2, y2 = (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))
    if x2 <= 0 or y2 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
        return None
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]



def _sim_distortion(img, annot, distortion):
    distorted_img = _distort_img(img, distortion)
    distorted_annots = _distort_bboxes(distorted_img, annot, distortion)
    return distorted_img, distorted_annots
    
def _distort_bbox(bbox, img, distortion):
    points = np.array([[[bbox[0], bbox[1]]], 
              [[bbox[2], bbox[1]]],
              [[bbox[0], bbox[3]]], 
              [[bbox[2], bbox[3]]]], dtype=np.float32)
    if len(img.shape) == 2:
        w, h = img.shape
    else:
        w, h, c = img.shape
    intrinsics = np.array([[h,0,int(h/2)],[0,w,int(w/2)],[0,0,1]])
    new_points = cv2.undistortPoints(points, intrinsics, np.array([distortion,0,0,0]),None,intrinsics)
    new_points = _bounding_box(new_points.squeeze(),img)

    return new_points

def _distort_bboxes(img, annot, distortion):
    new_annot = None
    for i in range(len(annot)):
        bbox = annot[i][...,:4]
        label = annot[i][...,4]
        distorted_bbox = _distort_bbox(bbox, img, distortion)
        if distorted_bbox is None:
            continue
        
        if new_annot is None:
            new_annot = np.concatenate([distorted_bbox, [label]])[None]
        else:
            new_annot = np.vstack((new_annot,np.concatenate([distorted_bbox, [label]])))
    if new_annot is None:
        new_annot = np.zeros((0,5),dtype=annot.dtype)
    return new_annot

def _distort_img(img, distortion):
    if len(img.shape) == 2:
        w, h = img.shape
    else:
        w, h, c = img.shape
    intrinsics = np.array([[h,0,int(h/2)],[0,w,int(w/2)],[0,0,1]])
    img = cv2.undistort(img, intrinsics, np.array([distortion,0,0,0]), newCameraMatrix=intrinsics)

    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Debug image file")
    parser.add_argument("path", type=str)

    opt = parser.parse_args()

    print(f"Load file {opt.path}")
    img = cv2.imread(opt.path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
      

    seperator = np.ones((img.shape[0],10,3),dtype=np.uint8)
    seperator[...,0] *= 90
    seperator[...,1] *= 30
    seperator[...,2] *= 150
    w,h = img.shape[:2]

    k = -0.2
    
    bboxes = np.array([[0,0,80,80, 0],[160,0,80,80, 1],[320,0,80,80, 2],[480,0,80,80, 3],
                        [80,160,80,80, 4],[240,160,80,80, 5],[400,160,80,80, 6],[560,160,80,80, 7],
                        [0,320,80,80, 8],[160,320,80,80, 9],[320,320,80,80, 10],[480,320,80,80, 11],
                        [80,400,80,80, 12],[240,400,80,80, 13],[400,400,80,80, 14],[560,400,80,80, 15],])

    bboxes[:,2:4] = bboxes[:,:2] + bboxes[:,2:4]
    dist, dist_boxes = _sim_distortion(img, bboxes, k)

    for bbox, dist_box in zip(bboxes,dist_boxes):
        bbox = [int(x) for x in bbox]
        dist_box = [int(x) for x in dist_box]
        img = cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:4]), (0,255,0), 2)
        dist = cv2.rectangle(dist, tuple(dist_box[:2]),tuple(dist_box[2:4]), (0,255,0), 2)


    concat = np.concatenate((img, seperator,dist), axis=1)

    cv2.imshow("img distorted", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
