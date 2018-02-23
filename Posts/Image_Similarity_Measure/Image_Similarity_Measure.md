---
layout: default
---

## Advertisement Similarity Detection

#### Problem Definition: 
Below is illustration of some duplicate advertisements which should be detected. (from now on I will use __item__ instead of __advertisement__)
The task is completely unsupervised and we have no labeled data of any similar items. 

<center>

An Item | A Duplicate
------------ | -------------
![Image of shoe](shoe.jpg) | ![Duplicate image of shoe](shoe_fc2.jpg)
![Image of hair](hair.jpg) | ![Duplicate Image of hair](hair_fc2.jpg)

</center>

Existence of similar items and showing them in the
same time to the user will harm advertising system's quality and
effectiveness.

#### A Naive Method:
To find similar image a trivial idea is to compute
L2 distance between images and if the distance is lower than
some threshold mark them as duplicates. This trivial method will cause
pool results because by changing point of view, objects position, and image contrast
the L2 distance will change significantly.
Below figures shows the closest image w.r.t Euclidean distance between two image.

<center>

An Item | Closest Item
------------ | -------------
![Image of shoe](shoe.jpg) | ![closest image to shoe](shoe_fc2.jpg)
![Image of hair](hair.jpg) | ![closest image to hair](hair_primary.jpg)

</center>

We see that L2 distance between images can produce rubbish results!
As you see closest image to top view of a head is a bride and broom
walking on a green field! (Perhaps because both images have a white area in the center.)

So can we do better? If we could transform each image to an space which closer points 
have similar characteristics then we can use L2 distance in transformed space and expect
better results.

#### Transforming Image Into meaningful space:
The idea is to use some pretrained neural network! I have used VGG19.
This network was trained on ImageNet dataset which consists of nearly 14 million images
which are labeled among 1000 different categories.
Some of these images are shown below

<center>

![Imagenet dataset](ImageNet.png)

</center>

VGG19 is one of the networks that was trained on this dataset to find category of 
each image. This network consists of some convolutional layers and 2 fully connected 
layer at the end, and a softmax layer to predict probabilities on each class. 

<center>

![vgg19 network](vgg19.jpg)

</center>

This network takes an image and do some transformations on that and goes forward to
reach the last layer which is a probability distribution on 1000 classes. As the
network learns to find the correct class for each images it learns to transform 
images into meaningful spaces. I reshaped each image to (300 * x), maintaining it's aspect
ratio and then center cropped each to obtain a (224 * 224 * 3) image.(3 relates to "RGB" channel)
VGG network takes an image and do some operations on it. input image is a (224 * 224 * 3)
tensor and for example output of the last max pooling layer is (7 * 7 * 512) tensor.
Two last fully connected layers are vector of 4096 elements, it means that each image
is transformed into a 4096 dimensional space at last fully connected layers.

So for each image we have output of different layers in VGG19 network which each regards
to a transformation of image. I have used last three layers (max_pool5, FC1, FC2) which are
(7 * 7 * 512), (4096), (4096) dimensional vectors.

Can we expect that using L2 distance in these new spaces work better than normal L2 distance
between images? The answer is __Yes__

Below tabel shows the closest images to shoe, hair, and car pictures, in terms of L2 Distance in
__primary, max_pool5, FC1, FC2__ spaces.

<center>

Space | Shoe Item | Hair Item |  Car Item
------------ | ------------ | ------------- | -------------
Image | ![Image of shoe](shoe.jpg) | ![Image of hair](hair.jpg) | ![Image of car](car.jpg)
Closest in Primary space | ![closest to shoe in primary space](shoe_fc2.jpg) | ![closest to hair in primary space](hair_primary.jpg) | ![closest to car in primary space](car_primary.jpg)
Closest in Pool5 space| ![closest to shoe in Max_pool5 space](shoe_fc2.jpg) | ![closest image to hair in Max_pool5 space](hair_pool5.jpg) | ![closest image to hair in Max_pool5 space](car_pool5.jpg)
Closest in FC1 space| ![closest to shoe in FC1 space](shoe_fc2.jpg) | ![closest image to hair in FC1 space](hair_fc1.jpg) |  ![closest image to hair in FC1 space](car_fc1.jpg)
Closest in FC2 space| ![closest to shoe in FC2 space](shoe_fc2.jpg) | ![closest image to hair in FC2 space](hair_fc2.jpg) | ![closest image to hair in FC2 space](car_fc2.jpg)

</center>

As you can see distances in FC2 space is meaningful and images which have lower distance 
are related from human point of view! (Images which have lower distance in primary space
are related from pixels point of view!)

[back](./)
