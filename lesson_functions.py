import numpy as np
import cv2
from skimage.feature import hog

def convert_color(image, cspace='BGR'):
	if cspace != 'BGR':
		if cspace == 'HSV':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		elif cspace == 'LUV':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
		elif cspace == 'HLS':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
		elif cspace == 'YUV':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		elif cspace == 'YCrCb':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
		elif cspace == 'Gray':
			feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else: feature_image = np.copy(image)
	return feature_image

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
						vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), 
								  transform_sqrt=False, 
								  visualise=vis, feature_vector=feature_vec, block_norm='L2')
		return features, hog_image
	# Otherwise call with one output
	else:	  
		features = hog(img, orientations=orient, 
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block), 
					   transform_sqrt=False, 
					   visualise=vis, feature_vector=feature_vec, block_norm='L2')
		return features

def bin_spatial(img, size=(32, 32), colorspace='BGR', hog_channel=0):
	if colorspace == 'Gray':
		return cv2.resize(img, size).ravel()
	elif hog_channel == 'ALL':
		color1 = cv2.resize(img[:,:,0], size).ravel()
		color2 = cv2.resize(img[:,:,1], size).ravel()
		color3 = cv2.resize(img[:,:,2], size).ravel()
		return np.hstack((color1, color2, color3))
	else:
		return cv2.resize(img[:,:,hog_channel], size).ravel()
						
def color_hist(img, nbins=32, cspace='BGR', hog_channel=0, bins_range=(0, 256)):
	if cspace == 'Gray':
		channel1_hist = np.histogram(img, bins=nbins, range=bins_range)
		return channel1_hist[0]
	elif hog_channel == 'ALL':
		# Compute the histogram of the color channels separately
		channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
		channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
		channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
		return hist_features
	else:
		channel1_hist = np.histogram(img[:,:,hog_channel], bins=nbins, range=bins_range)
		return channel1_hist[0]
	
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='BGR', orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32),
						hist_bins=32):							
				
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		# Read in each one by one
		image = cv2.imread(file)
		feature_image = convert_color(image, cspace) 
		# Call get_hog_features() with vis=False, feature_vec=True
		if cspace == 'Gray':
			hog_features = get_hog_features(feature_image, orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		elif hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)		
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# Apply bin_spatial() to get spatial color features
		spatial_features = bin_spatial(feature_image, spatial_size, cspace, hog_channel)
		# Apply color_hist() also with a color space option now
		hist_features = color_hist(feature_image, hist_bins, cspace, hog_channel)
		# Append the new feature vector to the features list
		features.append(np.concatenate((spatial_features, hist_features, hog_features)))
	# Return list of feature vectors
	return features
