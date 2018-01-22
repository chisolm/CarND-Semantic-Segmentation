# Semantic Segmentation

---

[//]: # (Image References)

[arch1]: ./writeup_images/deconv_layers.png "Deconvolution layers"
[loss1]: ./writeup_images/cross_entropy.png "Cross entropy loss"

[gi1]: ./writeup_images/um_000003.png "RR, seperated road, cars"
[gi2]: ./writeup_images/um_000015.png "Small road, oncoming cars"
[gi3]: ./writeup_images/umm_000093.png "Off Ramp particle weight"
[gi4]: ./writeup_images/uu_000001.png "Narrow road, buildings, parked cars"

[bi1]: ./writeup_images/uu_000001.png "Shoulder pulloff"
[bi2]: ./writeup_images/uu_000043.png "Paved shoulder"
[bi3]: ./writeup_images/uu_000094.png "Unusual failure, brick shoulder"

### Introduction

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

The approach of this project is to use a portion of a pre-trained network
VGG-16, the convolution layers 1-7 and the fully connected layers as the
convolution portion of the FCN.  

As described in the project plan, this architecture uses 1x1 convolutions 
from the last fully connected layer, and the 3rd and 4th convolution layers.  

The 1x1 outputs are processed through a deconvolution layer and then added
to the later layer and next deconvolution unit.  The picture below should
provide better information:

![alt text][arch1]

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Run

Run the following command to run the project:
```
python main.py train
```

By default with the 'train' action it will train the network and save
the inference results.  It will not save the model by default. (There is a --save option to save the model for later use.)

To use a saved model to process images, use the 'predict' action.  You can
also specify an alternate image location:
```
python main.py --model_dir logs/model18.01.21.21\:49/ predict
```

To use a stored model to process a video, you can do something like this with 
using default input and output locations:

```
python main.py --model_dir logs/model18.01.21.21\:49/ predict_movie
```

Complete usage with command line defaults is available with -h/--help option:

```
usage: main.py [-h] [--epochs EPOCHS] [--save] [--no-save] [--batch BATCH]
               [--keep_prob KEEP_PROB] [--learning_rate LEARNING_RATE]
               [--model_dir MODEL_DIR] [--training_data TRAINING_DATA]
               [--inference_dir INFERENCE_DIR]
               [--input_image_pat INPUT_IMAGE_PAT] [--video_input VIDEO_INPUT]
               [--video_output VIDEO_OUTPUT]
               {train,predict,predict_movie}

Process flags.

positional arguments:
  {train,predict,predict_movie}
                        train or predict on saved model: [train | predict]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to run
  --save
  --no-save
  --batch BATCH         Batch size. default: 128
  --keep_prob KEEP_PROB
                        Keep probability. default: 0.5
  --learning_rate LEARNING_RATE
                        Learning rate. default: 0.00005
  --model_dir MODEL_DIR
                        Dir to save model. default: logs/model
  --training_data TRAINING_DATA
                        training images dir, default: data/data_road/training
  --inference_dir INFERENCE_DIR
                        testing images dir, default: data/data_road/testing
  --input_image_pat INPUT_IMAGE_PAT
                        input images for prediction, as pattern default:
                        data/data_road/testing/image_2/*.png
  --video_input VIDEO_INPUT
                        File for video input. default: project_video.mp4
  --video_output VIDEO_OUTPUT
                        File name for video output. default:
                        project_video_output.mp4

```

# Rubric Points

## Project Construction Rubric points

This project meets all the project construction Rubric points:

### Does the project load the pretrained vgg model?

Complete

### Does the project learn the correct features from the images?

Complete

### Does the project optimize the neural network?

Complete

### Does the project train the neural network?

Complete.  The loss is printed at each step.

![alt text][loss1]

## Neural Network Training

### Does the project train the model correctly?

The model decreases loss consistently through epoch approximately epoch
26-28 with the default values.  THe curve and inconsistency at that point
suggest it may continue to improve with a reduced learning rate.

### Does the project use reasonable hyperparameters?

I use the following values by default:
```
    --epochs, default=30
    --batch, default=10
    --keep_prob, default=0.5
    --learning_rate, default=0.00005
```

I did test values around the given values for epoch, batch and learning rate.
Epoch and batch size seem reasonable.  The learning rate could use additional
investigation.

### Does the project correctly label the road?

I attempted to implemented an Intersection over Union metric for this project
with 2 different methods.  I was not able to debug the failures in either
method given the time I had for the project.  This would have given a much 
better measurement of the effectiveness of the project.

Visual review of the project shows that it does well in most scenarios.  The
following four images are taken for the variety of scenes they represent 
showing good performance.

![alt text][gi1]
![alt text][gi2]
![alt text][gi3]
![alt text][gi4]

The project had trouble with certain areas, such as certain shoulders, some
gravel or paved shoulders and normally able to distinguish brick shoulders
but not in this last case.

![alt text][bi1]
![alt text][bi2]
![alt text][bi3]

 I ran the netowrk against a video from the Advanced lane line project.  It
 did reasonably well at identifying the road surface, good performance against
 scenery and cars.  It did not do well with the inside shoulder and the
 freeway impact barrier.  The impact barrier performance was not surprising at
 all since it was not present in the training set.  The training set was also
 entirely european roads, so the generalization to data from US freeway data 
 was surprisingly successful.  The sample of this is included in the repository
 as project_video_output.mp4.



