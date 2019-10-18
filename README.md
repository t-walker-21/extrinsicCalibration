# extrinsicCalibration

scripts for calculating the rigid 3D transformation from the cv2 charUco/checkerboard board to the camera frame

REQUIREMENTS:

run pip install -r requirements.txt to install dependecies (numpy and opencv-contrib-python)

USAGE: 

FOR RUNNING CALIBRATION WITH CHARUCO BOARD, RUN:

python extrinsic_calib_charuco.py -i azureImages/ -k azure1_intrinsics.npz -r

FOR RUNNING CALIBRATION WITH CHECKERBOARD, RUN:

python extrinsic_calib_checkerboard.py -i azureCheckerboardImages/rgb/ -d azureCheckerboardImages/depth/
