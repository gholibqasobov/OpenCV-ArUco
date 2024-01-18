import cv2
import numpy as np
import pickle

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        marker_positions = dict()
        for(markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            marker_positions[markerID] = corners
            # print('corners:', corners)
            # print('marker pos:', marker_positions[markerID][0])
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                        thickness=2)
            # cv.putText(img, 'Hello', (255, 255), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

            if set([0, 1, 2, 3]).issubset(marker_positions.keys()):
                rect_coordinates = np.array([marker_positions[0], marker_positions[1],
                                             marker_positions[2], marker_positions[3]], np.float32)
                rect_coordinates = rect_coordinates.reshape((-1, 1, 2))
                # print('rectangle coordinates', rect_coordinates)
                rect_corners = np.array([marker_positions[0][2], marker_positions[1][3],
                                         marker_positions[2][0], marker_positions[3][1]], np.float32)
                rect_corners = rect_corners.reshape((-1, 1, 2))

                # print('rect corners: ', rect_corners)
                cv2.polylines(image, [rect_corners], isClosed=True, color=(0, 255, 0), thickness=2)

                """Open the detected area in a new window"""
                # we must detect the longest height and width and open the window of these parameters
                """
                bottom right = marker_positions[0][2][1] # the y coordinates
                
                
                
                height = (top left - bottom left) if max((top left - bottom left),(top right - bottom right)) ==
                 (top left - bottom left) else (top right - bottom right)
                width = (top right - top left) if max((top right - top left), (bottom right - bottom left ) == 
                (top left - top right) else (bottom right - bottom left)  
                """
                # we must open the window from the given points
                """
                
                """

    return image


# Loading the camera calibration parameters from pickle files
with open("C:\Projects\Robot Arm control with Computer Vision\CameraCalibration\calibration.pkl", "rb") as f:
    camera_matrx, dis_coeffs = pickle.load(f)


aruco_type = "DICT_4X4_100"
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width*(h/w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    detected_markers = aruco_display(corners, ids, rejected, img)

    cv2.imshow("Detected Rectangle", img)
    # cv2.imshow("Workplace Area", detected_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
