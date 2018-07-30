import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from camera_calibration import calibrate_camera
from lane_find_funcs import color_and_grad_binary, set_percpective_transform, warp_image
from lane_find_funcs import curv_calc_sliding_window, curv_calc_from_previous
from lane_find_funcs import add_lines_to_img, measure_curvature_and_offset

#%% 1st step - calibrate the camera
cal_values = calibrate_camera()
M, Minv = set_percpective_transform(img=cv2.imread('test_images/straight_lines1.jpg'), draw_intermidiate=False)

#%% When analyzing a video we can use the information of previous frames and use it to smooth out the outputs
# we can create a Line class that will keep track of this info
class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_fits = [] 
        self.recent_fits_cr = []
        self.last_output = [None]
        self.last_output_cr = [None]
 
    def process_line(self, fit, fit_cr):
        '''
        gets new line parameters, compares them to what was previously calculated and sends back updated lines accordingally
        '''
        if fit[0] == None:
            # if the algo didn't find any fit. we assume that is is a bad frame, and return the last value we have seen
            return self.last_output, self.last_output_cr
        
        if len(self.recent_fits) < 3:
            # if we don't have significant history, we just return the line as it is
            self.recent_fits.append(fit)
            self.recent_fits_cr.append(fit_cr)
            self.last_output, self.last_output_cr = fit, fit_cr
            return fit, fit_cr
        
        # calculate average of previous lines
        prev_fit_avg = np.mean(self.recent_fits, axis=0)
        prev_fit_cr_avg = np.mean(self.recent_fits_cr, axis=0)
        
        # calculate the L1 distace between the previous fits average and current line
        # if the difference is too big, we will discart the fit, assuming that it is a mistake
        # this scale should balance the effects of the weights
        scale = np.array([1e4,1e1,1e-2]) 
        l1_dist = np.sum(np.abs(fit*scale - prev_fit_avg*scale))
        if l1_dist > 10: # found this TH after some experimentation with the test images            
            return self.last_output, self.last_output_cr
            
        # smooth the output line, with previous linesaverage
        out_fit = 0.6 * fit + 0.4 * prev_fit_avg
        out_fit_cr = 0.6 * fit_cr + 0.4 * prev_fit_cr_avg
        
        # updated the fit history
        self.recent_fits.append(fit)
        self.recent_fits_cr.append(fit_cr)
        self.last_output, self.last_output_cr = fit, fit_cr
        
        # we keep only the last line fits for our calculations
        if len(self.recent_fits) > 6:
            self.recent_fits.pop(0)
        
        return out_fit, out_fit_cr
    
#%% new pipeline for videos:

# first we initialize the line objects
left_line = Line()
right_line = Line()

first_frame = True
left_fit = [None] # added as a safty condition for failures on previous frames

def video_frame_pipeline(img):
    global cal_values, M, Minv, left_line, right_line, first_frame, left_fit
    # step1 - undistort the image
    img = cv2.undistort(img, cal_values['mtx'], cal_values['dist'], None, cal_values['mtx'])
    
    # step2 - turn into binary image using color and gradient thresholds
    binary = color_and_grad_binary(img, s_thresh=(150,255), l_thresh=(50,255), sx_thresh=(100,255))
    
    # step3 - use perspective warping to look at the image from eagle eye view
    warped = warp_image(binary, M)
    warped = warped[100:,:] # mask the top part of the binary mask
    # step4 - find the pixels of the lanes and fit a polynom for them
    if first_frame:
        left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_sliding_window(warped)
        first_frame = False
    else:
        prev_left_fit = left_line.last_output
        prev_right_fit = right_line.last_output
        if left_fit[0] == None:
            # if the previous fit did not find a line, we try to recalculate it with sliding window
            left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_sliding_window(warped)
        try:
            left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_from_previous(warped, prev_left_fit, prev_right_fit)
        except:
            pass
        if left_fit[0] == None:
            # if the current fit did not find a line, we try to recalculate it with sliding window
            left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_sliding_window(warped)
        
    # step5 - process the lines by comparing them to the line history
    left_fit, left_fit_cr = left_line.process_line(left_fit, left_fit_cr)
    right_fit, right_fit_cr = right_line.process_line(right_fit, right_fit_cr)
    
    if left_fit[0] == None or right_fit[0] == None:
        return img # if after everything, we still failed to find a line, we return the original image...
    
    # step6 - unwarp the lines and draw them back on the original image
    lines_img = add_lines_to_img(img, left_fit, right_fit, Minv)
    
    # step7 - caculate the offset and curvature in real world units ([m])
    out = measure_curvature_and_offset(lines_img, left_fit_cr, right_fit_cr)
    return out

video_output = 'output_images/projectVideo.mp4'

clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(video_frame_pipeline)