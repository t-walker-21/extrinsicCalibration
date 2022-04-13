import numpy as np
import cv2 as cv
import os
import argparse
from sklearn.neighbors import NearestNeighbors
import transforms3d


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
        '-i', '--img_dir', required=True, help='Image dir for extrinsic calibration'
    )

arg_parser.add_argument(
        '-d', '--depth_img_dir', required=True, help='depth image dir for extrinsic calibration error calculation'
    )


args = vars(arg_parser.parse_args())



def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=30, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        distances = [np.linalg.norm(src[:m,:].T[i] - dst[:m,:].T[i]) for i in range(src.shape[1])]
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,:].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i



def compute3d(points_2d, depth, fx, fy, cx, cy, window=1):

    points_3d = np.empty((points_2d.shape[0], 3))

    unit_scaling = 0.001  # mm to m
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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

square_size = 0.02315

#azure
#mtx = np.array([[1925.18017,0,2035.44519],[0,1924.677978,1563.8046875],[0,0,1]])
#dist = np.array([0.427577,-2.7131388,0.0004280602,0.000448393,1.6005129,0.311038737,-2.536047,1.600512981])


#realsense - dist coeff are not provided, pass in zeros
mtx = np.array([[617.0289198,0,422.6674499],[0,617.010437011,248.56015],[0,0,1]])
dist = np.zeros(5)


cx = mtx.flatten()[2]
cy = mtx.flatten()[5]

fx = mtx.flatten()[0]
fy = mtx.flatten()[4]

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp *= square_size # 23.15 mm
# Arrays to store object points and image points from all the images.


results = []

axisLen = square_size * 2
axis = np.float32([[axisLen,0,0], [0,axisLen,0], [0,0,-axisLen]]).reshape(-1,3)

for fname in os.listdir(args['img_dir']):
    #print fname
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img = cv.imread(args['img_dir']+fname)
    img_raw = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    try:
        #print (args['depth_img_dir']+fname.replace(".png",".npy"))
        depth = np.load(args['depth_img_dir']+fname.replace(".png",".npy"))
        

    except:
        print ("no corresponding depth img for: ", fname)

    #cv.imshow(fname,cv.resize(gray,(500,500)))
    #cv.waitKey(1000)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        
        imgp = np.array(imgpoints).reshape(-1,2)
        _,rvec,tvec = cv.solvePnP(objp, imgp, mtx, dist)
        imgpts, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',cv.resize(img,(900,700)))
        cv.waitKey(0)


        #project image points into 3D

        pts3D = compute3d(imgp.astype(int),depth,fx,fy,cx,cy,window=10)
        
        rotMat = cv.Rodrigues(rvec)[0]

        T = np.zeros((4,4))
        T[:3, :3] = rotMat
        T[:3, 3] = tvec[:, 0]
        T[3,3] = 1

        T = np.linalg.inv(T)

        #transform points from camera frame to extrinsic board frame (inverse T)
        boardFramePoints = []

        errorAcc = []


        for p in range(len(objp)):

            boardFramePoints.append(np.dot(T, np.concatenate([pts3D[p], [1]]))[:3])

        

        boardFramePoints = np.array(boardFramePoints)



        error = boardFramePoints - objp

        #print "GT poitns: " , pts3D
        #print "calc Points: " , boardFramePoints

        #print "std dev: " , np.std(error,axis=1)


        #calculate euclidean distances between depth points and ground truth points 

        euc = np.linalg.norm(error,axis=1)

        #print "error: " , error

        #print "error_by_axis: " , np.mean(error,axis=0)
        #print "stddev_by_axis: " , np.std(error,axis=0)


        results.append(euc)

        




cv.destroyAllWindows()

results = np.array(results)


print ("averaged:")

print (np.mean(np.mean(results,axis=1)))

print ("std")

print (np.std(np.std(results,axis=1)))

print('done')