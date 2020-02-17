import numpy as np
from line_class import *
from image_func import *

def main():
	# Calculate important variables and save (keep global)
	# for each frame in a video:
	# 	undistort the image
	# 	apply thresholds (sobel, color etc.)
	# 	mask image
	# 	apply perspective transform
	# 	if reset == 1:
	# 		fitting a polynomial from scratch
	# 		take histogram of bottom half
	# 		find peaks to get starting position of left and right lane line
	# 		divide the image into n windows
	# 		for each window:
	# 			identify centerpoints of all lane pixels in each window
	# 			append the centerpoints to a list
	# 			fit a polynomial in y for these points
	# 			center point of this window becomes base of next window?
	# 		if lane line run to the side of the image:
	# 			reset = 1
	# 		else:
	# 			reset = 0
	# 	elif reset == 0:
	# 		use the previously fitted points and search around those points

	test_path = '../test_images'
	cal_path = '../camera_cal'
	img_size, mtx, dist, M, Minv = get_processing_param(test_path, cal_path)
	reset = 1
	left_line = Line()
	right_line = Line()
	lookahead_limit = 10
	lookahead_count = 0

	for frame in video:
		img = frame
		undist = cv2.undistort(img, mtx, dist, None, mtx)
		
		sobel_binary = abs_sobel_thresh(undist, 'x', 5, (20,90))
		s_binary = s_channel_thresh(undist, (150,255))
		combined_binary = np.zeros_like(s_binary)
		combined_binary[(s_binary==1) | (sobel_binary==1)] = 1
		color_binary = np.dstack(( np.zeros_like(s_binary), sobel_binary, s_binary)) * 255

		masked = mask_image(combined_binary)

		warped = apply_transform(masked, M, img_size)

		if reset == 1:
			left_line.xprev, left_line.yprev, right_line.xprev, right_line.y_prev = left_line.x, left_line.y, right_line.x, right_line.y
			left_line.x, left_line.y, right_line.x, right_line.y, out_img, ret = find_lane_pixels2(warped)
			if ret == 1:
				left_line.x, left_line.y, right_line.x, right_line.y = left_line.xprev, left_line.yprev, right_line.xprev, right_line.y_prev
				reset = 1
			out_img, ploty, left_line.fit, right_line.fit = fit_polynomial(warped, left_line.x, left_line.y, right_line.x, right_line.y, out_img)
			left_line.curverad, right_line.curverad = measure_curvature_pixels(warped, left_line.x, left_line.y, right_line.x, right_line.y, out_img)
			draw_lane_lines(warped, left_line.fit, right_line.fit, Minv, undist)
			if ret == 0 and 
		elif reset == 0:
			search_around_poly(binary_warped, left_line.fit, left_line.fit)

if __name__ == '__main__':
	main()