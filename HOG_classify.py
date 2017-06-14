import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from lesson_functions import *

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
colorspace = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb, Gray
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size=(32, 32)
hist_bins= 32

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

# Save SVM to be able to use it without having to train it everytime
# Adapted from https://github.com/ksakmann/CarND-Vehicle-Detection/blob/master/HOG_Classify.py
pickle_file = 'svc_pickle.p'
print('Saving data to pickle file...')
try:
	with open(pickle_file, 'wb') as pfile:
		pickle.dump(
			{   'svc':svc, 
				'scaler': X_scaler,
				'spatial_size': spatial_size,
				'hist_bins': hist_bins,
				'orient': orient,
				'pix_per_cell': pix_per_cell,
				'cell_per_block': cell_per_block,
				'colorspace': colorspace,
				'hog_channel': hog_channel
			},
			pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise

print('Data cached in pickle file.')

"""
# Call our function with vis=True to see an image output
img = cv2.imread(cars[0])
features, hog_image = get_hog_features(img[:,:,0], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
						
input_folder = './test_images/'
write_name = input_folder + 'output_car.jpg'
cv2.imwrite(write_name, img)
final_image_RGB = np.dstack((hog_image, hog_image, hog_image))*255
write_name = input_folder + 'output_visualization.jpg'
cv2.imwrite(write_name, final_image_RGB)
"""