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

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_CUBIC)


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
                # rect_coordinates = np.array([marker_positions[0], marker_positions[1],
                #                              marker_positions[2], marker_positions[3]], np.float32)
                # rect_coordinates = rect_coordinates.reshape((-1, 1, 2))

                rect_corners = np.array([marker_positions[0][2], marker_positions[1][3],
                                         marker_positions[2][0], marker_positions[3][1]], np.int32)
                rect_corners = rect_corners.reshape((-1, 1, 2))

                """workplace frame"""
                print('rect corners: ', rect_corners)
                # print('rect coords:', rect_coordinates)
                print('marker pos1', marker_positions[0][2])
                print('marker pos2', marker_positions[3][1])
                # print('marker pos-x1', marker_positions[0][2][0])
                # print('marker pos-

                # horizontal start point
                x = int(max(marker_positions[0][2][0], marker_positions[3][1][0]))
                print('max x', x)

                # width
                l1 = abs(marker_positions[1][3][0] - marker_positions[0][2][0])
                l2 = abs(marker_positions[2][0][0] - marker_positions[3][1][0])
                workplace_frame_w = int(min(l1, l2))
                print(workplace_frame_w)


                # vectical start point
                y = int(max(marker_positions[0][2][1], marker_positions[1][3][1]))
                print('y', y)

                # height of the frame
                h1 = abs(marker_positions[3][1][1] - marker_positions[0][2][1])
                h2 = abs(marker_positions[2][0][1] - marker_positions[1][3][1])
                workplace_frame_h = int(min(h1, h2))
                print(workplace_frame_h)

                # create the frame
                if workplace_frame_h is not None and workplace_frame_w is not None:
                    wokrplace_frame = img[y:y+workplace_frame_h, x:x+workplace_frame_w]
                    cv2.imshow('workplace', rescaleFrame(wokrplace_frame, 5))
                    # cv2.imshow('workplace', wokrplace_frame)
                cv2.polylines(image, [rect_corners], isClosed=True, color=(240, 207, 137), thickness=2)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
