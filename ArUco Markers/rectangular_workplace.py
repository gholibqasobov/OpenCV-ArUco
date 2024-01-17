import cv2
import numpy as np

camera_matrix = np.array([fx, 0, cx], [0, fy, cy], [0, 0, 1])
dist_coeffs = np.arr