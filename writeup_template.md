##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/original_car.jpg
[image11]: ./output_images/original_notcar.jpg
[image2]: ./output_images/output_car.jpg
[image21]: ./output_images/output_hog_car_visualization.jpg
[image22]: ./output_images/output_notcar.jpg
[image23]: ./output_images/output_hog_notcar_visualization.jpg
[image4]: ./output_images/output_test1.jpg
[image41]: ./output_images/output_test2.jpg
[image42]: ./output_images/output_test3.jpg
[image5]: ./output_images/output_7.jpg
[image51]: ./output_images/output_heatmap_7.jpg
[image52]: ./output_images/output_8.jpg
[image53]: ./output_images/output_heatmap_8.jpg
[image54]: ./output_images/output_9.jpg
[image55]: ./output_images/output_heatmap_9.jpg
[image7]: ./output_images/output_image_9.jpg
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `HOG_classify.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image11]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orient=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2] ![alt text][image21] ![alt text][image22] ![alt text][image23]

This file uses an auxiliary file called `lesson_functions.py` which is almost a copy of the one used in the lessons, but with some changes to adjust it to use OpenCV and other minor adaptions. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and after testing different color spaces, decided to use `BGR`, as it had the best results for me, despite of reading in the forums that most people used `YCrCb`. Then, I realized that I did a mistake when extracting the HOG features, as I was using a [0,1] range instead of [0,255] which is what OpenCV uses.

After correcting this bug, the performance increased extraordinarily (as expected) and I saw that `YUV` color space was which gave me the best results.

I tested different values for the orientation, pixels per cell etc. parameters using an automated test contained in the file `research.py` and examining the results decided to use the actual ones.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is done in the `HOG_classify.py` file, in lines 38-65. I trained a linear SVM using the extracted features for cars and non-cars, splitted in training and testing sets by the 80%-20% rule. The model has an accuracy of 0.9938 with the testing images. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This part is almost a copy of the lesson 35. Trying different values between 0.5 and 2.0, the best results where obtained mixing outputs for the values 1.25 and 1.5, so this are the scales used. The window overlapping is not changed from the original code.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image41]
![alt text][image42]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image5] ![alt text][image51]
![alt text][image52] ![alt text][image53]
![alt text][image54] ![alt text][image55]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I first started training the model with the basic image set provided by Udacity. As said, I began using the `BGR` color space due to the bug in the `find_cars` function. The results weren't very bad but there were quite a lot of false positives and the blue rectangles in the output video where definitely too shaky. Resolving this failure put me in the final path.

Possible improvements would be to make a sanitary check in order to eliminate false positives which don't make sense.

The final video shows some very fast detections due to cars in the opposite direction, and some false positives also. 

