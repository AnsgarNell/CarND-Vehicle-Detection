import numpy as np
import pickle
import cv2
from lesson_functions import *
import os
import glob
from scipy.ndimage.measurements import label
from collections import deque
import matplotlib.pyplot as plt

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
colorspace = dist_pickle["colorspace"]
hog_channel = dist_pickle["hog_channel"]

def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	# Return updated heatmap
	return heatmap# Iterate through list of bboxes
	
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale):
	
	rectangles = []
	
	# apply color conversion if other than 'RGB'
	ctrans_tosearch = convert_color(img, colorspace)
	if colorspace != 'Gray':
		ctrans_tosearch = ctrans_tosearch[ystart:ystop,:,:]
	else:
		ctrans_tosearch = ctrans_tosearch[ystart:ystop,:]
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	if colorspace != 'Gray':
		if hog_channel == 'ALL':
			ch1 = ctrans_tosearch[:,:,0]
			ch2 = ctrans_tosearch[:,:,1]
			ch3 = ctrans_tosearch[:,:,2]
		else:
			ch1 = ctrans_tosearch[:,:,hog_channel]
	else:
		ch1 = ctrans_tosearch

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2
	
	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	
	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	if (colorspace != 'Gray') and (hog_channel == 'ALL'):
		hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			if (colorspace != 'Gray') and (hog_channel == 'ALL'):
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			else:
				hog_features = hog_feat1

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], spatial_size)
		  
			# Get color features
			spatial_features = bin_spatial(subimg, spatial_size, colorspace, hog_channel)
			hist_features = color_hist(subimg, hist_bins, colorspace, hog_channel)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))		
			test_prediction = svc.predict(test_features)
			
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
				
	return rectangles
	
def draw_rectangles(img, rectangles1, rectangles2):
	draw_img = np.copy(img)
	for rectangle in rectangles1:
		cv2.rectangle(draw_img,(rectangle[0][0], rectangle[0][1]),(rectangle[1][0],rectangle[1][1]),(255,0,0),6)
	for rectangle in rectangles2:
		cv2.rectangle(draw_img,(rectangle[0][0], rectangle[0][1]),(rectangle[1][0],rectangle[1][1]),(255,0,0),6)
	return draw_img
	
def pipeline(img):

	global heatmaps,i

	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	converted_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	rectangles_125 = find_cars(converted_image, ystart, ystop, 1.25)
	heat = add_heat(heat, rectangles_125)
	rectangles_15 = find_cars(converted_image, ystart, ystop, 1.5)
	heat = add_heat(heat, rectangles_15)
	"""
	# Visualize the heatmap when displaying	
	heatmap = np.clip(heat, 0, 255)
	plt.imshow(heatmap, cmap='hot')
	write_name = output_folder + 'output_heatmap_' + str(i) + '.jpg'
	plt.savefig(write_name)
	plt.clf()
	"""
	heatmaps.append(heat)
	avg_heat = np.zeros_like(img[:,:,0]).astype(np.float)
	for i in range(1, len(heatmaps)):
		avg_heat = avg_heat + (heatmaps[i-1])*i
	# Apply threshold to help remove false positives
	avg_heat = apply_threshold(avg_heat,22)
	
	# Find final boxes from heatmap using label function
	labels = label(avg_heat)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	"""
	converted_image = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
	write_name = output_folder + 'output_image_' + str(i) + '.jpg'
	cv2.imwrite(write_name, converted_image)
	"""
	return draw_img
	
ystart = 400
ystop = 656
i = 0

# Make a list of calibration images
output_folder = './output_images/'
images = glob.glob('test_images/' + '*.jpg')

for image in images:
	filename = os.path.basename(image)
	print('Processing file', filename)
	img = cv2.imread(image)
	rectangles_125 = find_cars(img, ystart, ystop, 1.25)
	rectangles_15 = find_cars(img, ystart, ystop, 1.5)
	out_img = draw_rectangles(img, rectangles_125, rectangles_15) 
	write_name = output_folder + 'output_' + filename
	cv2.imwrite(write_name, out_img)

heatmaps = deque(maxlen=10)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

white_output = 'project_video_result.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("project_video.mp4").subclip(38,39)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
