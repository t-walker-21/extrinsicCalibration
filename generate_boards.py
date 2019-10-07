import cv2
import numpy as np

##### Intrinsic
# Use a predefined dictionary
# and for a bunch of fiducial markers
intrinsic_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Generate a Chessboard pattern + ArUco pattern board
# Input params:
# Num. squares in X direction = 6
# Num. squares in Y direction = 8
# Chess board square length = 1 inch
# Aruco marker square length = 1/2 inch
# NOTE: Make sure the prints align to this "real-world" dimensions
intrinsic_num_x = 6
intrinsic_num_y = 8
intrinsic_checkerboard_width = 1.274 * 25.4 / 1000 # inch -> mm -> m
intrinsic_marker_width = intrinsic_checkerboard_width / 2
intrinsic_board = cv2.aruco.CharucoBoard_create(
    intrinsic_num_x, intrinsic_num_y, intrinsic_checkerboard_width, intrinsic_marker_width, intrinsic_dictionary)

# Number parameters
INTRINSIC_MAX_ARUCO_IDS = intrinsic_num_x * intrinsic_num_y // 2
INTRINSIC_MAX_CHARUCO_IDS = (intrinsic_num_x - 1) * (intrinsic_num_y - 1)

##### Extrinsic
# Use a predefined dictionary
# and for a bunch of fiducial markers
extrinsic_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Generate a Chessboard pattern + ArUco pattern board
# Input params:
# Num. squares in X direction = 4
# Num. squares in Y direction = 6
# Chess board square length = 9 inch
# Aruco marker square length = 9/2 inch
# NOTE: Make sure the prints align to this "real-world" dimensions
extrinsic_num_x = 6
extrinsic_num_y = 8
extrinsic_checkerboard_width = 1.131 * 25.4 / 1000 # inch -> mm -> m
extrinsic_marker_width = extrinsic_checkerboard_width / 2
extrinsic_board = cv2.aruco.CharucoBoard_create(
    extrinsic_num_x, extrinsic_num_y, extrinsic_checkerboard_width, extrinsic_marker_width, extrinsic_dictionary)

# Number parameters
EXTRINSIC_MAX_ARUCO_IDS = extrinsic_num_x * extrinsic_num_y // 2
EXTRINSIC_MAX_CHARUCO_IDS = (extrinsic_num_x - 1) * (extrinsic_num_y - 1)


def main():
    # Draw the intrinsic board on a canvas of
    # 600 x 800 px with some borders
    intrinsic_board_img = intrinsic_board.draw((600, 800), 10, 1)

    # Save the image
    cv2.imwrite('calib_intrinsic_board.png', intrinsic_board_img)

    # Draw the extrinsic board on a canvas of
    # 3000 x 5000 px with some borders
    extrinsic_board_img = extrinsic_board.draw((4000, 6000), 100, 1)

    # Save the image
    cv2.imwrite('calib_extrinsic_board.png', extrinsic_board_img)

if __name__ == '__main__':
    main()