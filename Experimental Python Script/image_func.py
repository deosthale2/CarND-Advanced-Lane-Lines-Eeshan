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

def get_processing_param(test_path, cal_path):
	base_img = mpimg.imread(test_path+'/test1.jpg')
	# image size: dim[0] is height and dim[1] is width
	img_size = (base_img.shape[1], base_img.shape[0])
	# calibration parameters
	mtx, dist = get_calibration_param()
	# Perspective Transform Matrix
	src = np.float32([[300, 650], [1000, 650], [760,500], [520,500]])
	dest = np.float32([[300, 650], [1000, 650], [1000,500], [300,500]])
	M = cv2.getPerspectiveTransform(src, dest)
	Minv = cv2.getPerspectiveTransform(dest, src)
	return img_size, mtx, dist, M, Minv

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

def find_lane_pixels(binary_warped):
	# Take a histogram of bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

	# Create an output image to draw on visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))

	# Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    img_border_relaxation = 100
    leftx_base = np.argmax(histogram[img_border_relaxation:midpoint]) + img_border_relaxation
    rightx_base = np.argmax(histogram[midpoint:-img_border_relaxation]) + midpoint

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
    minpix_loss_count_left = 0
    minpix_loss_count_right = 0
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
        else:
        	minpix_loss_count_left += 1

        if len(good_rightx) > minpix:
        	rightx_current = np.int(np.mean(good_rightx))
        else:
        	minpix_loss_count_right += 1

        leftx += list(good_leftx)
        lefty += list(good_lefty)
        rightx += list(good_rightx)
        righty += list(good_righty)

    if minpix_loss_count_left>minpix_loss_count_right:
    	ret = (minpix_loss_count_left>nwindows//2)
    else:
    	ret = (minpix_loss_count_right>nwindows//2)

    return leftx, lefty, rightx, righty, out_img, ret

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, ploty

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

def draw_lane_lines(warped, left_fit, right_fit, Minv, undist):
	ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img_size[0], img_size[1])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	plt.imshow(result)