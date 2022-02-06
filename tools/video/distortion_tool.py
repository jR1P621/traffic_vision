import numpy as np
import cv2 as cv
import glob

from numpy.lib.type_check import imag

TEST_IMAGE = 'test_data/calibration/distortion_test7.png'
BOARD_DIMS = (9, 6)  # (horizontal, vertical)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_DIMS[0] * BOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_DIMS[1], 0:BOARD_DIMS[0]].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob(TEST_IMAGE)
if not images:
    print('Calibration image not found.')
    quit()
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray,
                                            (BOARD_DIMS[1], BOARD_DIMS[0]),
                                            None, cv.CALIB_CB_ADAPTIVE_THRESH)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (BOARD_DIMS[1], BOARD_DIMS[0]), corners2,
                                 ret)
        cv.imshow('img', img)
        cv.waitKey(500)

if not objpoints:
    print('No chess board found in calibration image.')
    quit()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                  gray.shape[::-1], None, None)
img = cv.imread(TEST_IMAGE)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h),
#                                         5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
print(
    f"Distortion coefficients:\nk1: {dist[0][0]}\nk2: {dist[0][1]}\nk3: {dist[0][4]}"
)
print(f'[{dist[0][0]},{dist[0][1]},{dist[0][4]}]')
cv.imshow('calib', dst)
cv.waitKey()
cv.destroyAllWindows()
