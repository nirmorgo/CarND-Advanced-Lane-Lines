import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from camera_calibration import calibrate_camera
from lane_find_funcs import color_and_grad_binary, set_percpective_transform, warp_image
from lane_find_funcs import curv_calc_sliding_window, curv_calc_from_previous
from lane_find_funcs import add_lines_to_img, measure_curvature_and_offset
#%% 1st step - calibrate the camera
cal_values = calibrate_camera()
M, Minv = set_percpective_transform(img=cv2.imread('test_images/straight_lines1.jpg'), draw_intermidiate=False)

#%% read all test images
images = glob.glob('test_images/*.jpg')
a = []
for img in images:
    img = cv2.imread(img)
    img = cv2.undistort(img, cal_values['mtx'], cal_values['dist'], None, cal_values['mtx'])
    binary = color_and_grad_binary(img, s_thresh=(150,255), l_thresh=(50,255), sx_thresh=(100,255), draw_intermidiate=False)
    warped = warp_image(binary, M, draw_intermidiate=False)
    warped = warped[100:,:] # mask the top part of the binary mask
    left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_sliding_window(warped, draw_intermidiate=False)
    a.append(left_fit)
    #left_fit, right_fit, left_fit_cr, right_fit_cr = curv_calc_from_previous(warped, left_fit, right_fit, draw_intermidiate=False)
    lines_img = add_lines_to_img(img, left_fit, right_fit, Minv, draw_intermidiate=False)
    out = measure_curvature_and_offset(lines_img, left_fit_cr, right_fit_cr, draw_intermidiate=True)

b = np.array([1e4,1e1,1e-2])
a = a*b
#%%
e = []
for i in range(len(a)):
    c = a[3,:]
    d = a[i,:]
    e.append(np.sum(np.abs(c-d)))
