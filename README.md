# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Example Outputs
Here are examples of a sufficient vs. insufficient output from a trained network:

Sufficient Result          |  Insufficient Result
:-------------------------:|:-------------------------:
![Sufficient](./examples/sufficient_result.png)  |  ![Insufficient](./examples/insufficient_result.png)

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Why Layer 3, 4 and 7?
In `main.py`, you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

### Results
Training ...
EPOCH 0/20 ...
GPU Time: 58.966609000000005 s | Loss: 0.6573686599731445
EPOCH 1/20 ...
GPU Time: 111.361344 s | Loss: 0.24221472442150116
EPOCH 2/20 ...
GPU Time: 163.638121 s | Loss: 0.0569184385240078
EPOCH 3/20 ...
GPU Time: 215.969321 s | Loss: 0.16114342212677002
EPOCH 4/20 ...
GPU Time: 268.25227500000005 s | Loss: 0.09518186748027802
EPOCH 5/20 ...
GPU Time: 320.53797299999997 s | Loss: 0.18352198600769043
EPOCH 6/20 ...
GPU Time: 372.864202 s | Loss: 0.13033707439899445
EPOCH 7/20 ...
GPU Time: 425.07716500000004 s | Loss: 0.21884459257125854
EPOCH 8/20 ...
GPU Time: 477.024453 s | Loss: 0.1922832876443863
EPOCH 9/20 ...
GPU Time: 528.665786 s | Loss: 0.20320585370063782
EPOCH 10/20 ...
GPU Time: 580.342461 s | Loss: 0.08749467879533768
EPOCH 11/20 ...
GPU Time: 632.0043999999999 s | Loss: 0.07161828875541687
EPOCH 12/20 ...
GPU Time: 683.6609649999999 s | Loss: 0.10299577564001083
EPOCH 13/20 ...
GPU Time: 735.3014909999999 s | Loss: 0.06795984506607056
EPOCH 14/20 ...
GPU Time: 786.909805 s | Loss: 0.10042920708656311
EPOCH 15/20 ...
GPU Time: 838.505763 s | Loss: 0.06330586969852448
EPOCH 16/20 ...
GPU Time: 890.126666 s | Loss: 0.11726796627044678
EPOCH 17/20 ...
GPU Time: 941.843547 s | Loss: 0.09034546464681625
EPOCH 18/20 ...
GPU Time: 993.479199 s | Loss: 0.07777317613363266
EPOCH 19/20 ...
GPU Time: 1045.120208 s | Loss: 0.035696156322956085
Training Finished. Saving test images to: ./runs/1547863927.1221182

### Optional sections
Within `main.py`, there are a few optional sections you can also choose to implement, but are not required for the project.

1. Train and perform inference on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). Note that the `project_tests.py` is not currently set up to also unit test for this alternate dataset, and `helper.py` will also need alterations, along with changing `num_classes` and `input_shape` in `main.py`. Cityscapes is a much more extensive dataset, with segmentation of 30 different classes (compared to road vs. not road on KITTI) on either 5,000 finely annotated images or 20,000 coarsely annotated images.
2. Add image augmentation. You can use some of the augmentation techniques you may have used on Traffic Sign Classification or Behavioral Cloning, or look into additional methods for more robust training!
3. Apply the trained model to a video. This project only involves performing inference on a set of test images, but you can also try to utilize it on a full video.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
