#Tevon Walker
#script to get 3D location of calib board points and calculate error of ground truth board points and 3D points projected into board frame
import cv2
import argparse
import os
import glob
from generate_boards import extrinsic_dictionary as dictionary, extrinsic_board as board
from generate_boards import EXTRINSIC_MAX_ARUCO_IDS as MAX_ARUCO_IDS
from generate_boards import EXTRINSIC_MAX_CHARUCO_IDS as MAX_CHARUCO_IDS
import numpy as np
from utils.config import LoadConfig, SaveConfig



results = []


def get_args():
    # Setup argument parser
    # and parse the inputs
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-i', '--img_dir', required=True, help='Image directory for extrinsic calibration'
    )

    arg_parser.add_argument(
        '-k', '--intrinsic', required=True, help='File storing intrinsic parameters (output of find_intrinsics.py)'
    )

    arg_parser.add_argument(
        '-o', '--out-dir', help='Output file directory', default='.'
    )

    arg_parser.add_argument(
        '-n', '--name', help='File (prefix) to store calibration parameters in', default='calib'
    )

    arg_parser.add_argument(
        '-r', '--relaxed-conditions', help='Set this flag to indicate relaxed condtions for number of markers found on board', action='store_true'
    )

    args = vars(arg_parser.parse_args())

    return args



def compute3d(points_2d, depth, fx, fy, cx, cy, window=1):

    points_3d = np.empty((points_2d.shape[0], 3))

    unit_scaling = 1  # mm to m
    constant_x = unit_scaling / fx
    constant_y = unit_scaling / fy
    bad_point = float('nan')

    for ind, point in enumerate(points_2d):
        x = 0.
        y = 0.
        z = 0.
        n_points_used = 0

        for v in range(point[1] - window//2, point[1] + window//2 + 1):
            if v>=0 and v < depth.shape[0]:
                for u in range(point[0] - window//2, point[0] + window//2 + 1):
                    if u>=0 and u < depth.shape[1]:
                        pt_depth = depth[v, u]
                        if pt_depth > 0:
                            # Fill in XYZ
                            x += (u - cx) * pt_depth * constant_x
                            y += (v - cy) * pt_depth * constant_y
                            z += pt_depth * unit_scaling
                            n_points_used += 1

        if n_points_used > 0:
            points_3d[ind] = [x / n_points_used, y / n_points_used, z / n_points_used]
        else:

            points_3d[ind] = [bad_point, bad_point, bad_point]

    return points_3d


def main():

    args = get_args()

    intrinsics = LoadConfig(args['intrinsic']).load()

    intrinsics['dist_coeffs'] = np.float32(intrinsics['dist_coeffs'][0])



    #camera intrincs
    cx = intrinsics['camera_matrix'].flatten()[2]
    cy = intrinsics['camera_matrix'].flatten()[5]

    fx = intrinsics['camera_matrix'].flatten()[0]
    fy = intrinsics['camera_matrix'].flatten()[4]


    #get file filst
    f_list = os.listdir(args['img_dir']+"rgb/")
    file_list = [] #final list of files to evaluate on


    for i in f_list:
        if "_" in i or ".png" not in i: #do not append files that have already been marked
            continue

        file_list.append(i)


    


    for f in file_list: #loop through files in image dir and find depth projection errors

        #print f

        f_color = args['img_dir'] + "rgb/"+f
        f_depth = args['img_dir'] + "depth/"+f.replace(".png",".npy")


        img = cv2.imread(f_color)

        #get corresponding depth image (numpy array)

        imgDepth = np.load(f_depth)



        #undistort image
        
        img_un = cv2.undistort(img,intrinsics['camera_matrix'],intrinsics['dist_coeffs'])


        # cv2.imshow("undistort",img_un)
        # cv2.waitKey(0)


        img_raw = img.copy() #copy of image for reprojection visualization



        # Detect just the ArUco (fiducial) markers in the image with
        # the same dictionary we used to generate the board (imported)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
            img, dictionary)



        # Refine the detected markers to find the missing
        # IDs. Required when we have boards with lots of IDs.
        corners, ids = cv2.aruco.refineDetectedMarkers(
            img, board, corners, ids, rejected_img_points)[:2]

        # If any markers in the dictionary are found, go ahead to
        # find the chessboard around the markers. Also, draw
        # the detected markers back on to the image.
        conditions_met = False
        if ids is not None:
            img_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            num_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, board)

            if charuco_corners is not None:
                img_markers = cv2.aruco.drawDetectedCornersCharuco(
                    img_markers, charuco_corners, charuco_ids)

                if ids.shape[0] == MAX_ARUCO_IDS \
                        and num_charuco == MAX_CHARUCO_IDS:
                    conditions_met = True
        else:
            img_markers = img

        



        # Write the image with markers back into the same folder
        if conditions_met or args['relaxed_conditions']:
                   

            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, intrinsics['camera_matrix'], intrinsics['dist_coeffs'])

            if ret: #successful pose recovered

                # print('Found pose of board')
                # print('X = {}, Y = {}, Z = {}'.format(*tvec.flatten()))
                # print('Rotation matrix: ')
                # print(cv2.Rodrigues(rvec)[0])

                # TODO: Generalize for other kinds of images
                img_with_axis = cv2.aruco.drawAxis(
                    img, intrinsics['camera_matrix'], intrinsics['dist_coeffs'], rvec, tvec, 0.25)
                """cv2.imwrite(args['img'].replace(
                    '.png', '_with_origin.png'), img_with_axis)"""


                
                #get actual object points and corresponding image points
                obj_points = board.chessboardCorners.squeeze()
                image_points = charuco_corners.squeeze()


                for point in image_points: #visualize points
                    
                    cv2.circle(img_raw,(point[0],point[1]), 2, (0,0,255), -1)


                #project image points into 3D

                pts3D = compute3d(image_points.astype(int),imgDepth,fx,fy,cx,cy,window=5)
                
                rotMat = cv2.Rodrigues(rvec)[0]

                T = np.zeros((4,4))
                T[:3, :3] = rotMat
                T[:3, 3] = tvec[:, 0]
                T[3,3] = 1

                T = np.linalg.inv(T)

                #transform points from camera frame to extrinsic board frame (inverse T)
                boardFramePoints = []

                errorAcc = []


                for p in range(len(pts3D)):

                    boardFramePoints.append(np.dot(T, np.concatenate([pts3D[p], [1]]))[:3])

                

                boardFramePoints = np.array(boardFramePoints)



                error = boardFramePoints - obj_points 

                #calculate euclidean distances between depth points and ground truth points 

                euc = np.linalg.norm(error,axis=1)

                #print euc

                results.append(euc)
                
                #cv2.imshow("image_points",cv2.resize(img_raw,(1000,900)))
                #cv2.waitKey(1000)

                """cv2.imwrite((args['img_dir'] + "/reprojection/" + f).replace(
                    '.jpg', '_with_reprojection.jpg'), img_raw)"""



            else:
                print('Did not find pose for board')

    


if __name__ == '__main__':
    main()

    results = np.array(results)
    
    print ("averaged:")

    print (np.mean(np.mean(results,axis=1)))

    print ("std")

    print (np.std(np.std(results,axis=1)))


