import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibrate_camera(images_path='camera_cal', nx=9, ny=6, draw_intermediate=False):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    path = images_path + '/calibration*.jpg'
    images = glob.glob(path)
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            if draw_intermediate:
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(200)
    
    cv2.destroyAllWindows()
    
    # Calbrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cal_values = {}
    cal_values['mtx'], cal_values['dist'] = mtx, dist
    
    # see example of an udistorted image
    if draw_intermediate:
        img = cv2.imread(images[0])
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.savefig('output_images/camera_calibration.png')
    
    return cal_values