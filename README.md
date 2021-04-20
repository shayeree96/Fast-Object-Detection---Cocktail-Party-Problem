# Fast-Object-Detection---Cocktail-Party-Problem

A common task in the vision community is to search objects from a large image dataset or video stream frames, where those objects only exist in a small proportion of images. 

## Introduction 

In this work we create a novel approach to utilize the cocktail party effect of convolutional neural networks (CNNs) for fast search. The cocktail party effect of CNNs means the ability to recognize the semantic information from a mixture of images. Specifically, we train and test detection models using channel-wise concatenated image groups. The model will learn to weighted-sum each group to form a mixture of images and extract features from the mixture. The features are used to either denoise the objects from object-background mixture or recognize the objects from object-object mixture. With the proposed testing pipeline, we greatly accelerate the searching speed compared to regular detection methods. Furthermore, our methodology can be easily applied to video processing, detection over video frames, and classification.

## Motivation

Let's say we are have a video dataset, where we are trying to find or search for a certain object. The object might we missing in many of the time frames, but our inferencing pipeline on any trained detection module would still scan the images one after another. What if we could pass two images into the detection pipeline as a mixture and check if either of the frames have the object or not and only localize for the frames if the object is classified to be present in that frame. This would drastically save us time on inferencing and actually reduce it by half. But the challenge is to train a detector which recognize the objects from object-object mixture and also object-background mixture and yet maintain the precision and recall in classification and the mAP for localization. 

## Understanding the Pipeline :

We use the Single Shot Detection model as our baseline for detection. This model uses VGG-16 Backbone and for simplicity we haven't added the FPN neck into the pipeline yet.
We train our model by mixing the images using pixel wise addition of two images. Let's say we have a batch size of 10, then we split the batch into two of size 5 each, and combine the images from the split batches into one. Then effectively a batch size of 5 would be sent into training.

In the diagram below let's consider for a single image mixture in a batch by adding image 1 and image 2, we get two target outputs assigned to the anchor at the detected locations

![image](https://user-images.githubusercontent.com/54212099/115314823-faf25b80-a143-11eb-8f48-04e220460ba9.png)

However one thing to note is that, since we train our model on a mixture of images, our ground truth targets to be assigned to the anchors are two for each detection location. Then we effectively are getting two labels for the a single location.

In order to understand the classification precision and recall, our novel approach uses the following logic:

The model will give prediction F(P), the predicted class of F(P) is denoted as F(P)1 and F(P)2, (if class 1 is background , then F(P)1 = background, vise versa)

### Note:   
  1. F(P)- False Postive in general
  2. F(P)1 - False Postive wrt Image 1 groundtruth
  3. F(P)2 - False Postive wrt Image 2 groundtruth
  4. G1 - Groundtruth for Image 1
  5. G2 - Groundtruth for Image 2
  
### PseudoCode for custom precision and recall calculation : 
```        
Set False postitive = 0
   True positive = 0
For all feature pixels
         if F(P)1 belongs to G1 or F(P)2 belongs to G1, and G1 is not Background
                 True positive += 1
         if F(P)1 belongs to G2 or F(P)2 belongs to G2, and G2 is not Background
                 True positive += 1
         if F(p)1 is not G1 or G2 ,
                 False positive += 1
         if F(p)2 is not G1 or G2 ,
                 False positive += 1
end for
note: G1 and G2 is groudtruth for feature pixel, not for bounding boxes
Recall = number(True positives) / ( number(G) - number(G_BK))# G_BK - No of pixels that are assigned background as Ground Truth
precision  =1 -  number(False positive) / number (F(P))
```     

Once we further train our model with the mixture of images on Pascal VOC pretrained weights for SSD, for 25 epochs on eight NVIDIA's GeForce RTX 2080 gpus, and get the classification precision and recall at par with the baseline SSD model which is trained without the mixture of images. 
Now the classification recall is of the utmost importance to us since it denotes whether the target object is present in either of two images or not. If not then we discard those frames, otherwise we take those frames and on our trained classification model, we now pass the images one by one and find the localization mAP and recall. 

Our novel approach helps especially during inferencing, we take a batch and halve it and mix the images in the splits using pixel-wise addition ( same as training ) and check the classification recall, if the target object is detected in either of the images, then we pass the images one by one through our trained ssd model to get the bounding boxes. Otherwise we move on to next set of image mixtures

#### Set up environment

Please check the mmdetection requirements.txt for setting up the environment. The repo is linked in citations.
However following are some general instructions :
```
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install mmcv==0.6.2
sudo chown -R ( mention the path where you have cloned the repo) /
pip install -v -e .
```

#### Download VOC dataset
Navigate to the project root directory

Make a data folder:
```
mkdir data
cd data/
```
Download tar files of VOC and extract them:
```
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCdevkit_18-May-2011.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
If you encounted this error:
```
Connecting to host.robots.ox.ac.uk (host.robots.ox.ac.uk)|129.67.94.152|:8080... failed: Connection refused.
``` 
Run the following commands:
```
wget http://pjreddie.com/media/files/VOC2012test.tar
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xvf VOC2012test.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
```

## File Structure
  1. configs/ours - This contains all the various configurations we have incorporated to enable mixture of image training as an extension to SSD pipeline 
  ```  cocktail1_our_ssd300_voc0712.py
       cocktail2_our_ssd300_voc0712.py
       cocktail3_our_ssd300_voc0712.py
       cocktail4_our_ssd300_voc0712.py
       cocktail5_our_ssd300_voc0712.py
       cocktail6_our_ssd300_voc0712.py
       cocktail7_our_ssd300_voc0712.py
       cocktail8_our_ssd300_voc0712.py
   ```
  2. mmdet/models/detectors - Here we have incorporated the cocktail_single_stage.py as a new detector head to enable the inference for a mixture of images and hence reduce the inference inference time by half
  3. mmdet/models/dense_heads - Here we have incorporated the different heads for classification based on the variations of loss functions which include:
  ```   our_ssd_head.py
        our_ssd_head_loss2.py
        our_ssd_head_loss3.py
        our_ssd_head_loss4.py
   ```
## Training
Navigate to the project root directory

We have trained for a certain cocktail configuration.

Run:
```
python tools/train.py configs/ours/cocktail1_our_ssd300_voc0712.py 

```

## Evaluation
Navigate to the project root directory. Note that here we have run a certain configuration. This may vary based on which cocktail configuration and trained file we want to test on. 

Run:
```

python tools/test_cc.py configs/ours/test_cocktail1_our_ssd300_voc0712.py work_dirs/ssd300_voc0712/epoch_24.pth --out 1.pkl --eval mAP

```

## Results



## Acknowledgements
Our project is based on the following two github repository:

https://github.com/open-mmlab/mmdetection/

We would like to thank the contributors for providing us code resources

## Patent 

U.S. Patent Application No. 63/147449, filed February 9, 2021. Patent Pending

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.
