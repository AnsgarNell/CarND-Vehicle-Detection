import numpy as np
import pickle
import cv2
from lesson_functions import *
import os
import glob

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

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel):
	
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	
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
				cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(255,0,0),6) 
				
	return draw_img
	
ystart = 400
ystop = 656
scale = 1.5

# Make a list of calibration images
input_folder = './project_video/'
images = glob.glob(input_folder + 'filename*.jpg')
#input_folder = './test_images/'
#images = glob.glob(input_folder + 'test*.jpg')
for image in images:
	filename = os.path.basename(image)
	print('Processing file', filename)
	img = cv2.imread(image)
	out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)
	write_name = input_folder + 'output_' + filename
	cv2.imwrite(write_name, out_img)