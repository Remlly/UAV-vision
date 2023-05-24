# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:52:37 2023

@author: remlly
"""

import cv2 as cv
import numpy as np
import glob
from enum import Enum

class command(Enum):
    
    calibrate = 1
    detectAruco = 2
    send = 3

def undistort(frame, mtx, dist):
    h,  w = frame.shape[:2]
    newMtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w,h))
    frame = cv.undistort(frame, mtx, dist, None, newMtx)
    
    x, y, w, h = roi
    return frame[y:y+h, x:x+w], newMtx, roi




cam = cv.VideoCapture(1)

#load values
mtx = np.load("cam_matrix.npy")
dist = np.load("distortion_values.npy")


#arUco parameters
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
parameters =  cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Define the physical dimensions of the ArUco marker in centimeters
marker_size = 8.5/2  # cm/2 want anders klopt de meting niet!

# Define the 3D object points in the marker's local coordinate system
markerObjp = np.zeros((4, 3), dtype=np.float32)
markerObjp[0] = [-marker_size/2, -marker_size/2, 0]
markerObjp[1] = [marker_size/2, -marker_size/2, 0]
markerObjp[2] = [marker_size/2, marker_size/2, 0]
markerObjp[3] = [-marker_size/2, marker_size/2, 0]


run = True    #run application
undist = True #apply new new matrix to camera image
case = command.calibrate 

tvec = 0
rvec = 0

while (run): #hoofdprogramma


    
    
    match case:
        case command.calibrate:
            
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((6*9,3), np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            
            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            images = glob.glob('C:/Users/remll/Desktop/UAV/Camera calibration/*.jpg')
            
            
            for fname in images:
                img = cv.imread(fname)
            
                #cv.imshow('img', img)
                #cv.waitKey(500)
            
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (9,6), None)
                # If found, add object points, image points (after refining them)
                print(ret)
            
                if ret == True:
                   
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, (9,6), corners2, ret)
                    #cv.imshow('img', img)
                    #cv.waitKey(500)
            
            cv.destroyAllWindows()
            
            #Save new values
            np.save("cam_matrix", mtx)
            np.save("distortion_values", dist)
            #reload
            mtx = np.load("cam_matrix.npy")
            dist = np.load("distortion_values.npy")
            #go to detect state
            case = command.detectAruco
            
        case command.detectAruco: 
            check, frame, = cam.read()
            cv.imshow('video2', frame)
            if undist:
                frame, newMtx, roi = undistort(frame, mtx, dist)
            
            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
            key = cv.waitKey(1)
            
            frame = cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            
            for k in range(len(markerCorners)):
              retval, rvec, tvec = cv.solvePnP(markerObjp,markerCorners[k], newMtx, dist)
              frame = cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, 5);  
              #print(f"ID{markerIds}, T{tvec}")
            
            cv.imshow('video', frame)
            
            
            if key == 27:
                break
            
            case = command.send

        case command.send:
            
            rvec = np.round(rvec, 3)
            tvec = np.round(tvec, 3)
            print(tvec)
            case = command.detectAruco
    
cam.release()
cv.destroyAllWindows()


