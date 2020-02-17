import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_calibration_param(cal_path):
	# 9*6 Chessboard
	# 9 columns 6 rows (9 in x and 6 in y direction)
	# create a matrix for storing [x,y,z] coordinate for 54 points
	# assign x and y coordinates to be (0,0), (1,0), ..., (8,0), (0,1), ...
	
	nx, ny = 9, 6
	objp = np.zeros((nx*ny, 3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Save objectpoints (Chessboard coordinates) and imagepoints (actual coordinates)
	objpoints = []
	imgpoints = []

	images = glob.glob(cal_path+'/calibration*.jpg')

	test_image = mpimg.imread(images[0])
	img_size = (test_image.shape[1], test_image.shape[0])

	for fname in images:
		img = mpimg.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# find Chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If corners are found, add object and image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

	return mtx, dist

class ImageOps(object):
	"""docstring for ImageOps"""
	def __init__(self, base_img):
		self.base_img = base_img

	def get_processing_param(self):
		# image size: dim[0] is height and dim[1] is width
		self.img_size = (self.base_img.shape[1], self.base_img.shape[0])
		# calibration parameters
		self.mtx, self.dist = get_calibration_param()
		# Perspective Transform Matrix
		src = np.float32([[300, 650], [1000, 650], [760,500], [520,500]])
		dest = np.float32([[300, 650], [1000, 650], [1000,500], [300,500]])
		self.M = cv2.getPerspectiveTransform(src, dest)

	def tarnsform_image(img):
		warped = cv2.warpPerspective(img, self.M, self.img_size)
		return warped

	def apply_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0,255)):
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if orient == 'x':
			sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
			abs_sobel = np.absolute(sobel_x)
		if orient == 'y':
			sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
			abs_sobel = np.absolute(sobel_y)
		sobel_scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
		sobel_binary = np.zeros_like(sobel_scaled)
		sobel_binary[(sobel_scaled>=thresh[0]) & (sobel_scaled<=thresh[1])] = 1
		return sobel_binary

	def s_channel_thresh(img, thresh=(0,255)):
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])] = 1
		return s_binary

	def mask_img(img):
		mask = np.zeros_like(img)
		ignore_mask_color = 255
		imshape = img.shape
		vertices = np.array([[(165,imshape[0]),(435, 310), (600, 310), (imshape[1],imshape[0])]], dtype=np.int32)
		cv2.fillPoly(mask, vertices, ignore_mask_color)
		masked_img = cv2.bitwise_and(img ,mask)
		return masked_img

	def find_lane_pixel(binary_warped):
		# Take a histogram of bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

		# Create an output image to draw on visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))

		# Find the peak of the left and right halves of the histogram
	    # These will be the starting point for the left and right lines
	    midpoint = np.int(histogram.shape[0]//2)
	    leftx_base = np.argmax(histogram[:midpoint])
	    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	    # HYPERPARAMETERS
	    # Choose the number of sliding windows
	    nwindows = 9
	    # Set the width of the windows +/- margin
	    margin = 100
	    # Set minimum no.pixel to detect a lane
	    minpix = 50

	    # Set height of windows - based on nwindows above and image shape
	    window_height = np.int(binary_warped.shape[0]//nwindows)

	    # Identify the x and y positions of all nozero pixels in the image
	    nonzero = binary_warped.nonzero()
	    nonzeroy = np.array(nonzero[0])
	    nonzerox = np.array(nonzero[1])

	    # Current positions to be updated later for each window in nwindows
	    leftx_current = leftx_base
	    rightx_current = rightx_base

	    # Create empty lists to receive left and right lane pixel indices
	    left_lane_inds = []
	    right_lane_inds = []
	    leftx, lefty = [], []
	    rightx, righty =[], []

	    # Step through the windows one by one
	    for window in range(nwindows):
	    	win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    	win_y_high = binary_warped.shape[0] - window*window_height

	    	win_xleft_low = leftx_current - margin
	    	win_xleft_high = leftx_current + margin
	    	win_xright_low = rightx_current - margin
	    	win_xright_high = rightx_current + margin

	    	# Draw the windows on the visualization image
	        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
	        (win_xleft_high,win_y_high),(0,255,0), 2) 
	        cv2.rectangle(out_img,(win_xright_low,win_y_low),
	        (win_xright_high,win_y_high),(0,255,0), 2)

	        good_leftx = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high].nonzero()[1] + win_xleft_low
	        good_lefty = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high].nonzero()[0] + win_y_low
	        good_rightx = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high].nonzero()[1] + win_xright_low
	        good_righty = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high].nonzero()[0] + win_y_low

	        if len(good_leftx) > minpix:
	        	leftx_current = np.int(np.mean(good_leftx))
	        if len(good_rightx) > minpix:
	        	rightx_current = np.int(np.mean(good_rightx))

	        leftx += list(good_leftx)
	        lefty += list(good_lefty)
	        rightx += list(good_rightx)
	        righty += list(good_righty)

	    return leftx, lefty, rightx, righty, out_img

	def fit_polynomial(binary_warped, leftx, lefty, rightx, righty, out_img):
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
		try:
	        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	    except TypeError:
	        # Avoids an error if `left` and `right_fit` are still none or incorrect
	        print('The function failed to fit a line!')
	        left_fitx = 1*ploty**2 + 1*ploty
	        right_fitx = 1*ploty**2 + 1*ploty

	    ## Visualization ##
	    # Colors in the left and right lane regions
	    out_img[lefty, leftx] = [255, 0, 0]
	    out_img[righty, rightx] = [0, 0, 255]

	    # Plots the left and right polynomials on the lane lines
	    plt.plot(left_fitx, ploty, color='yellow')
	    plt.plot(right_fitx, ploty, color='yellow')

	    return out_img, ploty, left_fit, right_fit

	def measure_curvature_pixels(binary_warped, leftx, lefty, rightx, righty, out_img):
	    '''
	    Calculates the curvature of polynomial functions in pixels.
	    '''
	    # Start by generating our fake example data
	    # Make sure to feed in your real data instead in your project!
	    out_img, ploty, left_fit, right_fit = fit_polynomial(binary_warped, leftx, lefty, rightx, righty, out_img)
	    
	    # Define y-value where we want radius of curvature
	    # We'll choose the maximum y-value, corresponding to the bottom of the image
	    y_eval = np.max(ploty)
	    
	    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
	    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	    
	    return left_curverad, right_curverad