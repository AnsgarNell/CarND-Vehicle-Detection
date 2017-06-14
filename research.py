import numpy as np
import pickle
import cv2
from lesson_functions import *
import os
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Divide up into cars and notcars
images = glob.glob('images/**/*.png', recursive=True)
cars = []
notcars = []
for image in images:
	if 'non-vehicles' in image:
		notcars.append(image)
	else:
		cars.append(image)
		
### TODO: Tweak these parameters and see how the results change.
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size=(32, 32)
hist_bins=32

ystart = 400
ystop = 656
scale = 1.5

input_folder = './test_images/'
# Make a list of calibration images
images = glob.glob(input_folder + 'test*.jpg')

def test(colorspace):
	t=time.time()
	car_features = extract_features(cars, cspace=colorspace, orient=orient, 
							pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)
	notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
							pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to extract HOG features...')
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)						
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	
	return svc, X_scaler
	
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
	
def apply_model(images, colorspace, hog_channel, svc, X_scaler):
	for image in images:
		filename = os.path.basename(image)
		img = cv2.imread(image)
		out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)
		write_name = input_folder + 'output_' + colorspace + '_' + str(hog_channel) + '_' + filename
		cv2.imwrite(write_name, out_img)

colorspace = 'BGR' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'BGR' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'BGR' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'BGR' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)



colorspace = 'HSV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'HSV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'HSV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'HSV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)




colorspace = 'LUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'LUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'LUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'LUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)



colorspace = 'HLS' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'HLS' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'HLS' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'HLS' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)



colorspace = 'YUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'YUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'YUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'YUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)


colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)

colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	
colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)


colorspace = 'Gray' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
svc, X_scaler = test(colorspace)
apply_model(images, colorspace, hog_channel, svc, X_scaler)
	



