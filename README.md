# Analysis of the Object Detection Pipeline onboard UAVs
by Leon Amadeus Varga, Sebastian Koch and Andreas Zell

In this repository, you will find additional information for the same named publication.
Besided further details about the experiments and a full list of all runs, you will find a preprocessing pipeline tool. This tool can help you to test the effect of the parameters for your specific application.

## Cite as
```
TODO
```
## Preprocessing pipeline code
see code/

## Further details
This information is intended as an extension of the publication and is helpful if you need more implementation details. 
### EXPERIMENTS - Datasets
#### DOTA-2
Ding et al. provide high-resolution satellite images with annotations [1]. We used the recommended preprocessing technique to create 1024x1024 tiles of the high-resolution recordings. All the recordings are provided in the lossless PNG format. Some images contain JPEG compression artifacts, so we assume that not the entire dataset creation pipeline was lossless.

[1]: Jian Ding and Nan Xue and Gui-Song Xia and Xiang Bai and Wen Yang and Michael Ying Yang and Serge J. Belongie and Jiebo Luo and Mihai Datcu and Marcello Pelillo and Liangpei Zhang 2021. Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges. CoRR, abs/2102.12219.

#### SeaDronesSee-DET
SeaDronesSee focus on search and rescue in maritime environments [2]. The training set includes 2,975 images with 21,272 annotations. We reduced the maximum side length of the images to 1024 pixels for this data set.
The reported accuracy is evaluated on the test set (with 1,796 images). This data set uses the PNG format [3], which utilizes a lossless compression.

[2]: Varga, L., Kiefer, B., Messmer, M., and Zell, A. 2022. Seadronessee: A maritime benchmark for detecting humans in open water. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2260–2270).

[3]: Thomas Boutell 1997. PNG (Portable Network Graphics) Specification Version 1.0. RFC, 2083, p.1–102.

#### VisDrone-DET
Zhu et al. proposed one of the most prominent UAV recordings data set [4]. The focus of this data set is traffic surveillance in urban areas. We use the detection task (VisDrone-DET). The training set contains 6,471 images with 343,205 annotations. Unless otherwise mentioned, we reduced the maximum side length by down-scaling to 1024 pixels.
We used the validation set for evaluation. The validation set contains 548 images. All images of this data set are provided in JPEG format [5] with a compression quality of 95.

[4]: Pengfei Zhu and Longyin Wen and Dawei Du and Xiao Bian and Qinghua Hu and Haibin Ling 2020. Vision Meets Drones: Past, Present and Future. CoRR, abs/2001.06303.

[5]: Gregory K. Wallace 1991. The JPEG Still Picture Compression Standard. Commun. ACM, 34(4), p.30–44. 

### EXPERIMENTS - Models
#### CenterNet
Duan et al. proposed CenterNet [6], an anchor-free object detector. The network uses a heat-map to predict the center-points of the objects. Based on these center-points, the bounding boxes are regressed.
Hourglass-104 [7] is a representative for extensive backbones, while the ResNet-backbones [8] cover a variety of different backbone sizes.
The ResNet backbones were trained with Adam and a learning rate of 1e-4. Further, we also used the plateau learning scheduler. For the Hourglass104-backbone, we used the learning schedule proposed by Pailla et al. [9].

[6]: Kaiwen Duan and Song Bai and Lingxi Xie and Honggang Qi and Qingming Huang and Qi Tian 2019. CenterNet: Keypoint Triplets for Object Detection. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019 (pp. 6568–6577). IEEE.

[7]: Newell, A., Yang, K., and Deng, J. 2016. Stacked hourglass networks for human pose estimation. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (pp. 483–499).

[8]: He, K., Zhang, X., Ren, S., and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (pp. 770–778). IEEE Computer Society.

[9]: Dheeraj Reddy Pailla and Varghese Alex Kollerathu and Sai Saketh Chennamsetty 2019. Object detection on aerial imagery using CenterNet. CoRR, abs/1908.08244.

#### EfficientDet
EfficientDet is optimized for efficiency and can perform well with small backbones [10]. For our experiments, we used EfficientDet with two backbones. The d0-backbone is the smallest and fastest of this family. And the d4-backbone represents a good compromise between size and performance.
We used three anchor scales (0.6, 0.9, 1.2), which are optimized to detect the small objects of the data sets. For the optimization, we used an Adam optimizer [11] with a learning rate of 1e-4. Further, we used a learning rate scheduler, which reduces the learning rate on plateaus with patience of 3 epochs.

[10]: Mingxing Tan and Ruoming Pang and Quoc V. Le 2020. EfficientDet: Scalable and Efficient Object Detection. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020 (pp. 10778–10787). Computer Vision Foundation / IEEE.

[11]: Kingma, D., and Ba, J. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings.

#### Faster R-CNN
Faster R-CNN is the most famous two-stage object detector [12]. And many of its improvements achieve today still state-of-the-art results [13]. We use three backbones for Faster R-CNN. A ResNet50, a ResNet101 of the ResNet-family [14] and a ResNeXt101 backbone [15].
For Faster R-CNN, we use the Adam optimizer [11] with a learning rate of 1e-4 and a plateau scheduler.

[12]: Shaoqing Ren and Kaiming He and Ross B. Girshick and Jian Sun 2017. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Trans. Pattern Anal. Mach. Intell., 39(6), p.1137–1149.

[13]: Pengfei Zhu and Longyin Wen and Dawei Du and Xiao Bian and Qinghua Hu and Haibin Ling 2020. Vision Meets Drones: Past, Present and Future. CoRR, abs/2001.06303.

[14]: He, K., Zhang, X., Ren, S., and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (pp. 770–778). IEEE Computer Society.

[15]: Xie, S., Girshick, R., Dollár, P., Tu, Z., and He, K. 2017. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1492–1500).

#### Yolov4
Bochkovskiy et al. published YoloV4 [16], which is the latest member of the Yolo-family providing a scientific publication. Besides a comprehensive architecture and parameter search, they did an in-depth analysis of augmentation techniques, called 'bag of freebies', and introduced the Mosaic data augmentation technique.
YoloV4 is a prominent representative of the object detectors because of impressive results on MS COCO. By default, YoloV4 scales all input images down to an image size of 608x608 pixels. For our experiments, we removed this preprocessing to improve the prediction of smaller objects.

[16]: Bochkovskiy, A., Wang, C., and Liao, H.. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection.


### EXPERIMENTS - Hardware setup
#### Desktop environment
The models were trained on computing nodes equipped with four GeForce GTX 1080 Ti graphic cards. The system was based on the Nvidia driver (version 460.67), CUDA (version 11.2) and PyTorch (version 1.9.0). To evaluate the inference speed in a desktop environment, we used a single RTX 2080 Ti with the same driver configuration. 

#### Embedded environment
The Nvidia Xavier AGX provides 512 Volta GPU cores, which are the way to go for the excessive forward passes of the neural networks. Therefore, it is a flexible way to bring deep learning approaches into robotic systems. We used the Nvidia Jetpack SDK in the version 4.6, which is shipped with TensorRT 8.0.1. TensorRT can speed up the inference of trained models. It optimizes these for the specific system, making it a helpful tool for embedded environments. For all embedded experiments, the Xavier board was set to the power mode 'MAXN', which consumes around 30W and utilizes all eight CPUs. Further, it makes use of the maximum GPU clock rate of around 1377 MHz.

### RESULTS AND DISCUSSION - Resolution
The findings for the SeaDronesSee dataset also apply to the VisDrone dataset. Only for the training image size of 1024 pixels, this rule does not hold. This can be explained by VisDrone's validation dataset. It does not contain many high-resolution images. Only about 3.5% have a full HD resolution. Most of the images are smaller and have a resolution of 1360x765 pixels. Therefore, the advantage of higher resolutions cannot be fully used with for a validation size of 2048. The added Background pixels add another source of error.

### RESULTS AND DISCUSSION - Application
For the **) evaluations, we used a maximal side length of 1280 pixels. We used therefore the CroW technique [17] in the default configuration. No further changes were applied.

[17]: Varga, L., and Zell, A. 2021. Tackling the Background Bias in Sparse Object Detection via Cropped Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops (pp. 2768-2777).

## Full list of experiment runs
[Dota runs](https://wandb.ai/starfleet/project_cp_evaluation/reports/Runs-of-Dota-data-set--VmlldzoxNTg4NjQw?accessToken=7ljtva7fnj6d4f01eh3gv5rgxudasx8bp069564hqlx9dcmp0f1zghmpal69j3o3)

[SeaDronesSee runs](https://wandb.ai/starfleet/project_cp_evaluation/reports/Runs-of-SeaDronesSee-data-set--VmlldzoxNTg4ODkx?accessToken=7bvk6l7c891nhc5oj287ond0f3d0xiu9q5k58q366uykl4a4g1cw7m93l1bsv3zm)

[VisDrone runs](https://wandb.ai/starfleet/project_cp_evaluation/reports/Runs-of-VisDrone-data-set--VmlldzoxNTg4MzIx?accessToken=j1rl20twfeveln4g2vfmp3qt0eem4q416jxhl2xf16hhevo6ed8126jgdobmt2ez)





