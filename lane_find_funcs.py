import numpy as np
import cv2
import matplotlib.pyplot as plt

def color_and_grad_binary(img, s_thresh=(0,255), l_thresh=(0,255), sx_thresh=(20,100), draw_intermidiate=False):
    img = np.copy(img)
    # Convert to HLS color space and separate the channels
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold light channel - this channel is used to mask shaded areas
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # combine the thresholds with "or"
    binary = np.zeros_like(s_channel)
    binary[((s_binary == 1) | (sxbinary == 1)) & (l_binary == 1)]  = 1
    
    if draw_intermidiate:    
            # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=40)
        
        ax2.imshow(binary,cmap='gray')
        ax2.set_title('Binary Image', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('output_images/binary_image.png')
    return binary

def set_percpective_transform(img=None, draw_intermidiate=False):
    '''
    a function that returns the lane lines percpective transform matrixes
    '''
    # Vertices selected manually for performing a perspective transform
    img_size = (img.shape[1], img.shape[0])
    
    top_left = [(img_size[0] // 2) - 60, img_size[1] // 2 + 100]
    bottom_left = [(img_size[0] // 6) - 10, img_size[1]]
    bottom_right = [(img_size[0] * 5 // 6) + 60, img_size[1]]
    top_right = [(img_size[0] // 2 + 60), img_size[1] // 2 + 100]  
    source = np.float32([top_left,bottom_left,bottom_right,top_right])
    # save points in format that can later be used for plotting polygon on image
    pts = np.array([top_left,bottom_left,bottom_right,top_right], np.int32)
    pts = pts.reshape((-1,1,2))
    # Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
    top_left = [(img_size[0] // 4), 0]
    bottom_left = [(img_size[0] / 4), img_size[1]]
    bottom_right = [(img_size[0] * 3 / 4), img_size[1]]
    top_right = [(img_size[0] * 3 / 4), 0]
    dst = np.float32([top_left,bottom_left,bottom_right,top_right])
    
    M = cv2.getPerspectiveTransform(source, dst)
    M_inv = cv2.getPerspectiveTransform(dst, source)
    
    if draw_intermidiate:
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)    
        copy = img.copy()
        copy = cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(copy,cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('output_images/perspective_warp.png')
    return M, M_inv
    
def warp_image(img, M, draw_intermidiate=False):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR) 
    if draw_intermidiate:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('Warped Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return warped

def curv_calc_sliding_window(warped, draw_intermidiate=False):
    out_img = None
    # first we find the lines centers using a histogram
    bottom_half = warped[warped.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    if draw_intermidiate:
        # Create an output image to draw on and visualize the result
        plt.figure()
        out_img = np.dstack((warped, warped, warped))*255
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
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if draw_intermidiate:
            # Draw the windows on the visualization image
            out_img = cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            out_img = cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # if we don't have enough valid pixels, we return None
    if len(leftx) < 5 or len(rightx) < 5:
        return [None], [None], [None], [None]
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # calculate real world dimension fit
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    if draw_intermidiate:
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        plt.figure(figsize=(12,8))
        plt.imshow(out_img)
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.savefig('output_images/window_lane_find.png')

    return left_fit, right_fit, left_fit_cr, right_fit_cr


def curv_calc_from_previous(warped, left_fit, right_fit, draw_intermidiate=False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
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
    
    # if we don't have enough valid pixels, we return None
    if len(leftx) < 5 or len(rightx) < 5:
        return [None], [None], [None], [None]

    # Fit new polynomials
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # calculate real world dimension fit
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    if draw_intermidiate:
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped))*255
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
        plt.figure(figsize=(12,8))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.savefig('output_images/prev_poly_lane_find.png')
    ## End visualization steps ##
    
    return left_fit, right_fit, left_fit_cr, right_fit_cr

def add_lines_to_img(img, left_fit, right_fit, Minv, draw_intermidiate=False):
    
    ploty = np.linspace(0, img.shape[0]-1, num=img.shape[0])# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_line = np.array(np.transpose(np.vstack([left_fitx, ploty])))
    right_line = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
    line_points = np.vstack((left_line, right_line))
    lines = np.zeros_like(img)
    lines = cv2.fillPoly(lines, np.int_([line_points]), [0,255, 0])
    unwarped_lines = warp_image(lines, Minv)
    out = cv2.addWeighted(img, 1, unwarped_lines, 0.4, 0)
    
    if draw_intermidiate:
        plt.figure()
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.savefig('output_images/output.png')
    
    return out


def measure_curvature_and_offset(img, left_fit_cr, right_fit_cr, draw_intermidiate=False):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img.shape[0]
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # The offset would be the distance between the middle of the lanes at the screen bottom, and the middle of the image
    lanes_center = np.mean([left_fit_cr[0]*(y_eval*ym_per_pix)**2 + left_fit_cr[1]*y_eval*ym_per_pix + left_fit_cr[2],
                      right_fit_cr[0]*(y_eval*ym_per_pix)**2 + right_fit_cr[1]*y_eval*ym_per_pix + right_fit_cr[2]])
    offset = lanes_center - (img.shape[1] / 2) * xm_per_pix
    
    radius_text = "Radius=%.3fm" % (np.mean([left_curverad, right_curverad]))
    offset_text = "Offset=%.3fm" % offset
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.putText(img, radius_text, (10,40), font, 1, (255,255,255),2)
    out = cv2.putText(img, offset_text, (10,80), font, 1, (255,255,255),2)
    
    if draw_intermidiate:
        plt.figure(figsize=(12,8))
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.savefig('output_images/output.png')
    
    return out