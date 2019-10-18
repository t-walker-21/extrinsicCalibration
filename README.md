# extrinsicCalibration

scripts for calculating the rigid 3D transformation from the cv2 charUco/checkerboard board to the camera frame, and evaluating the error. Using openCV charuco and checkerboards and rgbd images, extrinsic calibration is evaluated. First a transform is calcuated using the checkerboard or charuco board. Next the corners from the board are projected from image frame to camera frame using the 3D value found at that pixel in the depth image and the camera intrinsics. Using these projected points and the transform from the board calibration, we transform the points from camera from to board frame, and compare the projected points seen by the camera to the ground-truth points that are defined by the board size and shape. This comparison shows that the two pointsets vary quite a bit. The Azure gives an average error of about 15 mm when using the charuco board and 12 mm when using the checkerboard pattern. The intel realsense 435D gives around 6.5 mm error.

REQUIREMENTS:

run pip install -r requirements.txt to install dependecies (numpy and opencv-contrib-python)

USAGE: 

FOR RUNNING CALIBRATION WITH CHARUCO BOARD, RUN:

python extrinsic_calib_charuco.py -i azureImages/ -k azure1_intrinsics.npz -r

FOR RUNNING CALIBRATION WITH CHECKERBOARD, RUN:

python extrinsic_calib_checkerboard.py -i azureCheckerboardImages/rgb/ -d azureCheckerboardImages/depth/

FOR RUNNING CALIBRATION WITH CHECKERBOARD IMAGES FROM REALSENSE D435, RUN:

python extrinsic_calib_checkerboard_realsense.py -i realSenseImages/rgb/ -d realSenseImages/depth/
