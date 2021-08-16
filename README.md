# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/augmented.png "Augmented Images"
[image2]: ./images/accuracy.png "Accuracy"
[image3]: ./images/gray.png "Gray Images"
[image4]: ./images/new.png  "Traffic signs"
[image5]: ./images/prob.png "Probilities of new images"
[image6]: ./images/signs.png "Random selection of traffic signs"
[image7]: ./images/testdist.png "Distribution of test set"
[image8]: ./images/traindist.png "Distribution of training set"
[image9]: ./images/validdist.png "Distribution of validation set"
[image10]: ./images/signoverview.png "Overview of traffic signs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 65598. The size of the origin training data is 34799. I augmented the training set by applying a rotation and brightness change of the origin training data. 
* The size of the validation set is 4410
* The size of test set is  12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

First, we start with an overview of all signs in the following figure:
![alt text][image10]

Also i selected random images to gain more insight about the images.
![alt text][image6]

From the exploration of the images the quality of the images varies strongly. There are good images, but also images with low brightness and/or sharpness. 
The next step was to explore the distribution of the classes of each image set. In general, the classes in each data set seems to be similar distributed. 
![alt text][image8]

![alt text][image9]

![alt text][image7]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing of the images include follow steps:


* Augmentation

To generate from the training set new images, a rotation to each image is applied by a random angle. The brightness of the images is also changed by a random value. This generates 34799 images. Combined with the original training set we have a total of 65598 images for training. The code for this step can be found in the function `augment_image`.
Here a few augmented images along with the original ones:

![alt text][image1]


* Convert to grayscale


My first approach was to use color images. During the training step, I observed that the training with grayscale images is a lot faster. Therefore I use the converted color images to grayscale.

* Normalization

Each grayscale image is normalized. The follow normalization is applied to each pixel: `(pixel - 128.) / 128`. 
This normalization scales each pixel between [-1,1]. This normalization prevents a slow or unstable learning process.


Here are some images after the preprocessing.

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


The model is based off LeNet.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flatten					|	outputs 400										|
|Fully connected					|	outputs 120											|
|Relu					|									|
|Dropout						|	keep probability=0.9											|
|Fully connected					|	outputs 84											|
|Relu					|									|
|Dropout						|	keep probability=0.9											|
|Fully connected					|	outputs 43											|
| Softmax				|      									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model the adam optimizer with a learning rate of 0.0008. I tried different learning rates in the range between 0.0005 and 0.001. This learning rate seems for this task the best option. With this learning rate a fast training is possible.

The batch size is set to 16. There are batch sizes in the range between 8 and 128 tested. One observation was that a smaller batch size increases the validation accuracy a lot. With the batch size of 16, we achieve a validation accuracy of 0.89 after the first epoch. A lower batch size results in a small improvement in accuracy, but the training for one epoch needs longer. Therefore, I used a batch size of 16.

The number of epochs was fixed with 35 epochs. I tried epochs up to 100, the validation accuracy increases but not much. 
During the training the validation accuracy fluctuating as we can see in this image:
![alt text][image2]

There may follow reasons for this fluctuation:
* learing rate to high: we take a too large step in the error landscape, with lower learning rates this improve but the training time increases a lot
* Overfitting: To prevent overfitting a dropout was used, but it may be useful to use an other kind of regularization e.g. L2 to penalize large weights



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


As a starting point for the model architecture, I started with LeNet. 
Since this model was developed for the recognition of handwritten letters, it works also for other images. The question was, which prediction accuracy can we achieve. The original LeNet model uses grayscale images. Our training set consists of color images. If we using in the input layer of the neural network a depth of 1 instead of 3, the training will be faster.

Therefore, I decided to use the same input structure as in the original model architecture proposed. The output layer is modified to support the 43 sign classes. 

The LeNet uses for the first two layers a convolutional layer.  I added a third convolutional layer, but there was not the expected improvement in the accuracy.  It seems two convolutional layers are sufficient for the correct recognition of traffic signs, which we will later see in the accuracy results. 

My final model results were:
* validation set accuracy of 0.957
* test set accuracy of 0.935

The model is changed to work with gra

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image5] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


