import numpy as np
import os
import cv2
from utils.transforms import linear_transform
from utils.transforms import fast_circle
from utils.midas import read_pfm

##------------------------------------------------------------------------------------
class AriaCamera:
    def __init__(self, cfg, human_id='', camera_name='', type_string='rgb', calibration_path='', images_path=''):
        self.cfg = cfg
        self.human_id = human_id
        self.type_string = type_string
        self.camera_type = 'ego'
        self.camera_name = camera_name

        self.calibration_path = calibration_path
        self.images_path = os.path.join(images_path, self.type_string)

        ## predictions from midas
        self.depth_path = images_path.replace('/ego/{}/images'.format(camera_name), '/processed_data/depth/{}'.format(camera_name))

        if type_string == 'rgb':
            self.undistorted_images_path = os.path.join(images_path, 'undistorted_rgb')

        self.rotated_images_path = self.images_path.replace(self.type_string, 'rotated_{}'.format(self.type_string)) ## these are the original VRS format images

        ###----------set image width and height, this is for the rotated images wrt to humans!
        self.image_height, self.image_width = self.set_image_resolution()
        self.rotated_image_height = self.image_width
        self.rotated_image_width = self.image_height

        self.intrinsics = None
        self.extrinsics = None

        return  

    ##--------------------------------------------------------
    def get_image_path(self, time_stamp):
        image_path = os.path.join(self.images_path, '{:05d}.jpg'.format(time_stamp))
        return image_path

    def get_rotated_image_path(self, time_stamp):
        rotated_image_path = os.path.join(self.rotated_images_path, '{:05d}.jpg'.format(time_stamp))
        return rotated_image_path

    ##--------------------------------------------------------
    def set_image_resolution(self):
        time_stamp_string = sorted(os.listdir(self.calibration_path))[0].replace('.txt', '')
        image_path = self.get_image_path(time_stamp=int(time_stamp_string))
        image = cv2.imread(image_path)
        image_height = image.shape[0]
        image_width = image.shape[1]
        return image_height, image_width

    ##--------------------------------------------------------
    def update(self, intrinsics, extrinsics):
        # the parameter vector has the following interpretation:
        # intrinsic = [f c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
        self.intrinsics = intrinsics
        assert(self.intrinsics.shape[0] == 15)

        self.extrinsics = extrinsics

        assert(self.extrinsics.shape[0] == 4 and self.extrinsics.shape[1] == 4)
        self.inv_extrinsics = np.linalg.inv(self.extrinsics)

        ## finding a XYX such that [0, 0, 0, 1] = [R, T; 0 1] * [XYZ1];
        center = np.dot(self.inv_extrinsics, np.array([0, 0, 0, 1]).reshape(4, 1))
        self.location = center[:3].reshape(-1)

        return  

    ##--------------------------------------------------------
    def get_depth_path(self, time_stamp):
        pfm_path = os.path.join(self.depth_path, '{:05d}.pfm'.format(time_stamp))
        return pfm_path

    def get_depth(self, time_stamp):
        pfm_path = os.path.join(self.depth_path, '{:05d}.pfm'.format(time_stamp))
        depth, scale = read_pfm(pfm_path)
        return depth

    ##--------------------------------------------------------
    def get_image(self, time_stamp):
        image_path = self.get_image_path(time_stamp=time_stamp)
        image = cv2.imread(image_path)
        return image

    def get_rotated_image(self, time_stamp):
        image_path = self.get_rotated_image_path(time_stamp=time_stamp)
        image = cv2.imread(image_path)
        return image

    def get_image_area(self):
        return self.image_width * self.image_height


    def init_undistort_map(self, resolution=1):
        # points_2d_x = np.arange(0, self.rotated_image_width, resolution).reshape(-1, 1)
        # points_2d_y = np.arange(0, self.rotated_image_height, resolution).reshape(-1, 1)
        
        # points_2d = np.concatenate([points_2d_x, points_2d_y], axis=1) ## 1408 x 2, rotated aria view
        # undistorted_points_2d = np.round(self.undistort(points_2d)).astype(np.int)
        # points_2d = np.round(points_2d).astype(np.int)

        # valid = (undistorted_points_2d[:, 0] >= 0) * (undistorted_points_2d[:, 0] < self.rotated_image_height) \
        #         * (undistorted_points_2d[:, 1] >= 0) * (undistorted_points_2d[:, 1] < self.rotated_image_width)

        # points_2d = points_2d[valid]
        # undistorted_points_2d = undistorted_points_2d[valid]

        # points_2d = self.get_inverse_rotated_pose2d(points_2d) ## go from aria frame to human frame
        # undistorted_points_2d = self.get_inverse_rotated_pose2d(undistorted_points_2d)        
        
        # self.undistorted_map = undistorted_points_2d
        # self.distorted_map = points_2d

        # ##------------above code is not working, we use opencv to undistort, this is an approximation!
        # f = self.intrinsics[0]
        # fx = f; fy = f
        # cx = self.intrinsics[1]
        # cy = self.intrinsics[2]
        # k1 = self.intrinsics[3]
        # k2 = self.intrinsics[4]
        # k3 = self.intrinsics[5]
        # k4 = self.intrinsics[6]
        # k5 = self.intrinsics[7]
        # k6 = self.intrinsics[8]
        # p1 = self.intrinsics[9]
        # p2 = self.intrinsics[10]
        # s1 = self.intrinsics[11]
        # s2 = self.intrinsics[12]
        # s3 = self.intrinsics[13]
        # s4 = self.intrinsics[14]
        # self.K_opencv = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # ## distCoeffs1 output vector of distortion coefficients [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy] of 4, 5, 8, 12 or 14 elements. 
        # self.D_opencv =  np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]) 
        # self.D_fisheye_opencv = np.array([k1, k2, k3, k4])
        # self.K_new_opencv, roi = cv2.getOptimalNewCameraMatrix(self.K_opencv, self.D_opencv, (self.rotated_image_width, self.rotated_image_height), 0)
        # self.mapx_opencv, self.mapy_opencv = cv2.initUndistortRectifyMap(self.K_opencv, self.D_opencv, None, self.K_new_opencv, (self.rotated_image_width, self.rotated_image_height), 5)

        ##----------using colmap fisheye----------
        colmap_info = '1 OPENCV_FISHEYE 1408 1408 610.28362376985785 612.90448786022057 704 704 0.40003139535703836 -0.55946670947205268 0.76795773277897017 -0.43835076136009982'
        colmap_info = [float(val) for val in colmap_info.split()[4:]]
        fx = colmap_info[0]
        fy = colmap_info[1]
        cx = colmap_info[2]
        cy = colmap_info[3]
        k1 = colmap_info[4]
        k2 = colmap_info[5]
        k3 = colmap_info[6]
        k4 = colmap_info[7]
        self.K_opencv = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.D_opencv = np.array([k1, k2, 0, 0, k3, k4, 0, 0]) ## distCoeffs1 output vector of distortion coefficients [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy] of 4, 5, 8, 12 or 14 elements. 
        self.D_fisheye_opencv = np.array([k1, k2, k3, k4,])


        ##--------------------------------------
        self.intrinsic_matrix = np.array([[self.intrinsics[0], 0, self.intrinsics[1]],
                            [0, self.intrinsics[0], self.intrinsics[2]],
                            [0 ,0 ,1]])

        self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix) 
        self.K = self.intrinsic_matrix

        ###--------cache the map-------------------
        cache_dir = self.calibration_path.replace('/calib', '')
        map_path = os.path.join(cache_dir, 'undistort_map.npz')

        if not os.path.exists(map_path):
            self.x_map = np.zeros((self.rotated_image_height, self.rotated_image_width), dtype=np.float32)
            self.y_map = np.zeros((self.rotated_image_height, self.rotated_image_width), dtype=np.float32)

            for x in range(self.rotated_image_width):
                for y in range(self.rotated_image_height):
                    cam_coords = self.intrinsic_matrix_inv @ np.array([x, y, 1]) # unproject from pinhole image to world coordinate
                    p = self.image_from_cam(cam_coords) # project camera coordinate to fisheye image
                    self.x_map[y][x] = p[0]
                    self.y_map[y][x] = p[1]

            print('saving undistort map to cache {}'.format(self.camera_name))
            np.savez(map_path, x_map=self.x_map, y_map=self.y_map)

        else:
            print('loading undistort map from cache {}'.format(self.camera_name))
            data = np.load(map_path)
            self.x_map = data['x_map']
            self.y_map = data['y_map']

        return

    def get_undistorted_image_aria(self, time_stamp):
        # https://github.com/facebookresearch/Aria_data_tools/issues/15
        image = self.get_image(time_stamp)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        undistorted_rotated_image = cv2.remap(rotated_image, self.x_map, self.y_map, cv2.INTER_CUBIC)
        undistorted_image = cv2.rotate(undistorted_rotated_image, cv2.ROTATE_90_CLOCKWISE)
        return undistorted_image


    def get_undistorted_image(self, time_stamp):
        image = self.get_undistorted_image_aria(time_stamp)
        return image

    def read_undistorted_image(self, time_stamp):
        image_path = os.path.join(self.images_path.replace('/rgb', '/undistorted_rgb'), '{:05d}.jpg'.format(time_stamp))
        image = cv2.imread(image_path)
        return image

    def undistort(self, batch_point_2d):
        assert(batch_point_2d.shape[0] >= 1 and batch_point_2d.shape[1] == 2)
        batch_undistorted_point_2d = []
        for point_2d in batch_point_2d:
            point_2d = self.undistort_(point_2d)
            batch_undistorted_point_2d.append(point_2d.reshape(1, -1))
        batch_undistorted_point_2d = np.concatenate(batch_undistorted_point_2d, axis=0)
        return batch_undistorted_point_2d

    def undistort_(self, point_2d):
        uvDistorted = (point_2d - self.intrinsics[1:3]) / self.intrinsics[0]
        xr_yr = self.compute_xr_yr_from_uvDistorted(uvDistorted)

        # early exit if point is in the center of the image
        xr_yrNorm = np.sqrt((xr_yr**2).sum())

        if xr_yrNorm == 0:
            temp_ = np.asarray([0, 0])
        else:
            theta = self.getThetaFromNorm_xr_yr(xr_yrNorm) ## is a double
            temp_ = (np.tan(theta) / xr_yrNorm) * xr_yr ## direct assignment to point_3d[:2] does not work!
        
        undistorted_point_2d = temp_ * self.intrinsics[0] + self.intrinsics[1:3]
        return undistorted_point_2d

    ##--------------------------------------------------------
    ## point_3d is N x 3
    def project_batch(self, batch_point_3d):
        assert(batch_point_3d.shape[0] > 1 and batch_point_3d.shape[1] == 3)
        batch_point_2d = []
        for point_3d in batch_point_3d:
            point_2d = self.project(point_3d=point_3d)
            batch_point_2d.append(point_2d.reshape(1, -1))
        batch_point_2d = np.concatenate(batch_point_2d, axis=0)
        return batch_point_2d


    ##--------------------------------------------------------
     ## 3D to 2D using perspective projection and then radial tangential thin prism distortion
    def project(self, point_3d):
        assert(point_3d.shape[0] == 3)
        point_3d_cam = self.cam_from_world(point_3d)
        point_2d = self.image_from_cam(point_3d_cam)

        return point_2d

    ## to camera coordinates 3D from world cooredinate 3D
    def cam_from_world(self, point_3d):
        assert(point_3d.shape[0] == 3)
        point_3d_homo = np.asarray([point_3d[0], point_3d[1], point_3d[2], 1])
        point_3d_cam = np.dot(self.extrinsics, point_3d_homo)
        point_3d_cam = point_3d_cam[:3]/point_3d_cam[3]
        return point_3d_cam

    ## reference: https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    def check_point_behind_camera(self, point_3d):    
        assert(point_3d.shape[0] == 3)
        return True if point_3d[2] < 0 else False

    ## to image coordinates 2D from camera cooredinate 3D
    ## reference: https://ghe.oculus-rep.com/minhpvo/NikosInternship/blob/04e672190a11c76d0b810ebc121f46a2aa5b67e4/aria_hand_obj/aria_utils/geometry.py#L29
    ## reference: https://www.internalfb.com/code/fbsource/[0c9e159d90aee8bfeff67a6e2066726a5ecc1796]/arvr/projects/surreal/experiments/PseudoGT/increAssoRecon/core/geometry.cpp?lines=567
    ## the intrinsics paramters are // intrinsic = [f_u {f_v} c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
    ## the intrinsics paramters are // intrinsic = [f c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}], 15 in total. f_u = f_v
    ## k_1, k_2, k_3, k_4, k_5, and k_6 are radial distortion coefficients. p_1 and p_2 are tangential distortion coefficients. Higher-order coefficients are not considered in OpenCV.
    def image_from_cam(self, point_3d, eps=1e-9):
        if self.check_point_behind_camera(point_3d):
            return np.array([-1, -1]) ## out of bound

        startK = 3
        numK = 6
        startP = startK + numK
        startS = startP + 2
        
        inv_z = 1/point_3d[-1]
        ab = point_3d[:2].copy() * inv_z ## makes it [u, v, w] to [u', v', 1]
        
        ab_squared = ab**2
        r_sq = ab_squared[0] + ab_squared[1]

        r = np.sqrt(r_sq)
        th = np.arctan(r)
        thetaSq = th**2

        th_radial = 1.0 
        theta2is = thetaSq

        ## radial distortion
        for i in range(numK):
            th_radial += theta2is * self.intrinsics[startK + i]
            theta2is *= thetaSq

        th_divr = 1 if r < eps else th / r

        xr_yr = (th_radial * th_divr) * ab
        xr_yr_squared_norm = (xr_yr**2).sum()

        uvDistorted = xr_yr
        temp = 2.0 * xr_yr * self.intrinsics[startP:startP+2]
        uvDistorted += temp * xr_yr + xr_yr_squared_norm * self.intrinsics[startP:startP+2]

        radialPowers2And4 = np.array([xr_yr_squared_norm, xr_yr_squared_norm**2])

        uvDistorted[0] += (self.intrinsics[startS:startS+2] * radialPowers2And4).sum()
        uvDistorted[1] += (self.intrinsics[startS+2:] * radialPowers2And4).sum()

        point_2d = self.intrinsics[0] * uvDistorted + self.intrinsics[1:3]

        return point_2d

    ##--------------------------------------------------------
    ## undistort, 2D to 3D using inverse of perspective projection
    def unproject(self, point_2d):
        point_3d_cam = self.cam_from_image(point_2d)
        point_3d_world = self.world_from_cam(point_3d_cam)
        return point_3d_world

    ## to world coordinates 3D from camera cooredinate 3D
    def world_from_cam(self, point_3d):
        assert(point_3d.shape[0] == 3)
        point_3d_homo = np.asarray([point_3d[0], point_3d[1], point_3d[2], 1])
        point_3d_world = np.dot(self.inv_extrinsics, point_3d_homo)
        point_3d_world = point_3d_world[:3]/point_3d_world[3]
        return point_3d_world

    ## to camera coordinates 3D from image cooredinate 2D
    ## reference: https://www.internalfb.com/code/fbsource/[0c9e159d90aee8bfeff67a6e2066726a5ecc1796]/arvr/projects/surreal/experiments/PseudoGT/increAssoRecon/core/geometry.cpp?lines=753
    def cam_from_image(self, point_2d):
        uvDistorted = (point_2d - self.intrinsics[1:3]) / self.intrinsics[0]

        xr_yr = self.compute_xr_yr_from_uvDistorted(uvDistorted)

        ## early exit if point is in the center of the image
        xr_yrNorm = np.sqrt((xr_yr**2).sum())

        if xr_yrNorm == 0:
            return np.asarray([0, 0, 1])

        theta = self.getThetaFromNorm_xr_yr(xr_yrNorm) ## is a double

        point_3d = np.ones(3) ## initialize
        point_3d[0] = 0; point_3d[1] = 0; point_3d[2] = 1 ## initialize

        temp_ = (np.tan(theta) / xr_yrNorm) * xr_yr ## direct assignment to point_3d[:2] does not work!
        point_3d[0] = temp_[0]
        point_3d[1] = temp_[1]

        return point_3d

    def compute_xr_yr_from_uvDistorted(self, uvDistorted, kMaxIterations=50, kDoubleTolerance2=1e-7*1e-7):
        startK = 3
        numK = 6
        startP = startK + numK
        startS = startP + 2

        xr_yr = uvDistorted # initialize

        for j in range(kMaxIterations):
            uvDistorted_est = xr_yr
            xr_yr_squared_norm = (xr_yr**2).sum()

            temp = 2 * xr_yr * self.intrinsics[startP:startP+2]
            uvDistorted_est += temp * xr_yr + xr_yr_squared_norm * self.intrinsics[startP:startP+2]

            radialPowers2And4 = np.array([xr_yr_squared_norm, xr_yr_squared_norm**2])

            uvDistorted_est[0] += (self.intrinsics[startS:startS+2] * radialPowers2And4).sum()
            uvDistorted_est[1] += (self.intrinsics[startS+2:] * radialPowers2And4).sum()

            ## compute the derivative of uvDistorted wrt xr_yr
            duvDistorted_dxryr =  self.compute_duvDistorted_dxryr(xr_yr, xr_yr_squared_norm)

            ## compute correction:
            ## the matrix duvDistorted_dxryr will be close to identity ?(for resonable values of tangenetial/thin prism distotions)
            ## so using an analytical inverse here is safe
            correction = np.dot(np.linalg.inv(duvDistorted_dxryr), uvDistorted - uvDistorted_est) 

            xr_yr += correction

            if (correction**2).sum() < kDoubleTolerance2:
                break

        return xr_yr

    ## helper function, computes the Jacobian of uvDistorted wrt the vector [x_r; y_r]
    def compute_duvDistorted_dxryr(self, xr_yr, xr_yr_squared_norm):
        startK = 3
        numK = 6
        startP = startK + numK
        startS = startP + 2

        duvDistorted_dxryr = np.zeros((2, 2)) ## initialize
        duvDistorted_dxryr[0, 0] = 1.0 + 6.0 * xr_yr[0] * self.intrinsics[startP] + 2.0 * xr_yr[1] * self.intrinsics[startP + 1]

        offdiag = 2.0 * (xr_yr[0] * self.intrinsics[startP + 1] + xr_yr[1] * self.intrinsics[startP])
        duvDistorted_dxryr[0, 1] = offdiag
        duvDistorted_dxryr[1, 0] = offdiag
        duvDistorted_dxryr[1, 1] = 1.0 + 6.0 * xr_yr[1] * self.intrinsics[startP + 1] + 2.0 * xr_yr[0] * self.intrinsics[startP]

        temp1 = 2.0 * (self.intrinsics[startS] + 2.0 * self.intrinsics[startS + 1] * xr_yr_squared_norm)
        duvDistorted_dxryr[0, 0] += xr_yr[0] * temp1
        duvDistorted_dxryr[0, 1] += xr_yr[1] * temp1

        temp2 = 2.0 * (self.intrinsics[startS + 2] + 2.0 * self.intrinsics[startS + 3] * xr_yr_squared_norm)
        duvDistorted_dxryr[1, 0] += xr_yr[0] * temp2
        duvDistorted_dxryr[1, 1] += xr_yr[1] * temp2

        return duvDistorted_dxryr

    ## helper function to compute the angle theta from the norm of the vector [x_r; y_r]
    def getThetaFromNorm_xr_yr(self, th_radialDesired, kMaxIterations=50, eps=1e-9):
        th = th_radialDesired
        startK = 3
        numK = 6
        startP = startK + numK
        startS = startP + 2

        for j in range(kMaxIterations):
            thetaSq = th*th

            th_radial = 1
            dthD_dth = 1

            ## compute the theta polynomial and its derivative wrt theta
            theta2is = thetaSq

            for i in range(numK):   
                th_radial += theta2is * self.intrinsics[startK + i]
                dthD_dth += (2*i + 3) * self.intrinsics[startK + i] * theta2is
                theta2is *= thetaSq

            th_radial *= th

            ## compute correction
            if np.abs(dthD_dth) > eps:
                step = (th_radialDesired - th_radial)/dthD_dth
            else:
                step = 10*eps if (th_radialDesired - th_radial)*dthD_dth > 0.0 else -10*eps

            th += step

            ## revert to within 180 degrees FOV to avoid numerical overflow
            if np.abs(th) >= np.pi / 2.0:
                ## the exact value we choose here is not really important, we will iterate again over it
                th = 0.999*np.pi/2.0

        return th

    ##--------------------------------------------------------
    ## return 3 x 3 rotation, world to cam
    def get_rotation(self):
        return self.extrinsics[:3, :3]

    ## return 3 x 1 translation, world to cam
    def get_translation(self):
        return self.extrinsics[:3, 3]

    def get_location(self):
        return self.location

    ##------------------------pose2d-------------------------
    ###----------human frame to aria frame----------
    def get_rotated_pose2d(self, pose2d, bbox=None):
        assert(pose2d.shape[1] == 3) ## x, y, score

        x = pose2d[:, 0].copy()
        y = pose2d[:, 1].copy()

        rotated_x = y
        rotated_y = self.image_width - x

        pose2d[:, 0] = rotated_x
        pose2d[:, 1] = rotated_y

        if bbox is not None:
            assert(bbox.shape[0] == 5) ## x, y, x, y, score

            x1, y1, x2, y2, score = bbox
            rotated_x1 = y1
            rotated_y1 = self.image_width - x1
            rotated_x2 = y2
            rotated_y2 = self.image_width - x2

            bbox = np.array([rotated_x1, rotated_y1, rotated_x2, rotated_y2, score]) 
            return pose2d, bbox

        return pose2d

    ###----------aria frame to human frame----------
    def get_inverse_rotated_pose2d(self, pose2d, bbox=None):
        assert(pose2d.shape[1] == 3 or pose2d.shape[1] == 2) ## x, y, score
        assert bbox is None

        x = pose2d[:, 0].copy()
        y = pose2d[:, 1].copy()

        rotated_x = self.rotated_image_height - y
        rotated_y = x 

        pose2d[:, 0] = rotated_x
        pose2d[:, 1] = rotated_y

        return pose2d

    ##------------------------bbox-------------------------
    ##-------------- bbox in aria frame----------------
    ## tagging = 0.02
    ## this works for VRS format true
    def get_rotated_bbox_2d(self, aria_human):
        bbox_thres = self.cfg.BBOX.EGO.MIN_AREA_RATIO
        close_bbox_distance = self.cfg.BBOX.EGO.CLOSE_BBOX_DISTANCE
        close_bbox_thres = self.cfg.BBOX.EGO.CLOSE_BBOX_MIN_AREA_RATIO
        max_aspect_ratio_thres = self.cfg.BBOX.EGO.MAX_ASPECT_RATIO
        min_aspect_ratio_thres = self.cfg.BBOX.EGO.MIN_ASPECT_RATIO

        bbox_3d = aria_human.get_bbox_3d()
        bbox_2d_all = self.project_batch(batch_point_3d=bbox_3d)
        bbox_2d_all = self.check_bounds_rotated(bbox_2d_all)

        ## the human is out of sight!
        if len(bbox_2d_all) == 0:
            return None       

        x1 = bbox_2d_all[:, 0].min()
        x2 = bbox_2d_all[:, 0].max()
        y1 = bbox_2d_all[:, 1].min()
        y2 = bbox_2d_all[:, 1].max()

        ## this is the reverse of actual image as this for the rotated image
        bbox_width = (x2 - x1)
        bbox_height = (y2 - y1)

        area = bbox_width * bbox_height
        image_area = self.get_image_area()
        bbox_area_ratio = area*1.0/image_area

        ## if bbox is too small
        if bbox_area_ratio < bbox_thres:
            return None

        distance_to_human = np.sqrt(((aria_human.location - self.location)**2).sum())

        ## the bbox is nearby and too small
        if distance_to_human < close_bbox_distance and bbox_area_ratio < close_bbox_thres:
            return None

        # print(self.camera_name, self.type_string, aria_human.human_name, area*1.0/image_area, distance_to_human)

        aspect_ratio = bbox_width/bbox_height ## height/width

        ## the bbox is too skewed, height is large compared to width
        if aspect_ratio > max_aspect_ratio_thres or aspect_ratio < min_aspect_ratio_thres:
            return None

        bbox_2d = np.array([round(x1), round(y1), round(x2), round(y2)]) ## convert to integers
        return bbox_2d

    ## check for points inside the image
    def check_bounds_rotated(self, bbox_2d_all):
        is_valid = (bbox_2d_all[:, 0] >= 0) * (bbox_2d_all[:, 0] <= self.rotated_image_width) * \
                    (bbox_2d_all[:, 1] >= 0) * (bbox_2d_all[:, 1] <= self.rotated_image_height)
        return bbox_2d_all[is_valid]

    ##--------get bbox in aria frame and then rotates it to human frame-------------
    def get_bbox_2d(self, aria_human):
        rotated_bbox_2d = None
        bbox_2d = self.get_rotated_bbox_2d(aria_human)

        if bbox_2d is not None:
            x1, y1, x2, y2 = bbox_2d
            rotated_x1 = self.rotated_image_height - y2
            rotated_y1 = x1
            rotated_x2 = self.rotated_image_height - y1
            rotated_y2 = x2
            rotated_bbox_2d = [rotated_x1, rotated_y1, rotated_x2, rotated_y2] ## swap x and y

        return rotated_bbox_2d


    ##------------get the location of the aria in 3d-----------
    def get_aria_location(self, point_3d):
        point_2d = self.project(point_3d)

        ##---aria to human frame
        if self.camera_type == 'ego':
            x = point_2d[0]
            y = point_2d[1]

            rotated_x = self.rotated_image_height - y
            rotated_y = x 

            point_2d = np.array([rotated_x, rotated_y])

        is_valid = self.is_inside_bound(point_2d)

        return point_2d, is_valid

    def is_inside_bound(self, point_2d):
        is_valid = (point_2d[0] >= 0) * (point_2d[0] < self.image_width) * \
                    (point_2d[1] >= 0) * (point_2d[1] < self.image_height) 
        return is_valid

    ##-------------vectorized version of projection----------------
    ## point_3d is N x 3, vectorized projection
     ## 3D to 2D using perspective projection and then radial tangential thin prism distortion
    def vec_project(self, point_3d):
        assert(point_3d.shape[0] >= 1 and point_3d.shape[1] == 3)
        point_3d_cam = self.vec_cam_from_world(point_3d)
        point_2d = self.vec_image_from_cam(point_3d_cam)
        return point_2d


    ##--------------------------------------------------------
    ## to camera coordinates 3D from world cooredinate 3D
    def vec_cam_from_world(self, point_3d):
        point_3d_cam = linear_transform(point_3d, self.extrinsics)
        return point_3d_cam

    ## reference: https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    def vec_check_point_behind_camera(self, point_3d):    
        return point_3d[:, 2] < 0 


    ## to image coordinates 2D from camera cooredinate 3D
    ## reference: https://ghe.oculus-rep.com/minhpvo/NikosInternship/blob/04e672190a11c76d0b810ebc121f46a2aa5b67e4/aria_hand_obj/aria_utils/geometry.py#L29
    ## reference: https://www.internalfb.com/code/fbsource/[0c9e159d90aee8bfeff67a6e2066726a5ecc1796]/arvr/projects/surreal/experiments/PseudoGT/increAssoRecon/core/geometry.cpp?lines=567
    ## the intrinsics paramters are // intrinsic = [f_u {f_v} c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
    ## the intrinsics paramters are // intrinsic = [f c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}], 15 in total. f_u = f_v
    ## k_1, k_2, k_3, k_4, k_5, and k_6 are radial distortion coefficients. p_1 and p_2 are tangential distortion coefficients. Higher-order coefficients are not considered in OpenCV.
    def vec_image_from_cam(self, point_3d, eps=1e-9):
        is_point_behind_camera = self.vec_check_point_behind_camera(point_3d)

        ##------------------
        startK = 3
        numK = 6
        startP = startK + numK
        startS = startP + 2

        ##------------------
        inv_z = 1/point_3d[:, -1]
        ab = point_3d[:, :2].copy() * inv_z.reshape(-1, 1) ## makes it [u, v, w] to [u', v', 1]
            
        ab_squared = ab**2
        r_sq = ab_squared[:, 0] + ab_squared[:, 1]
        
        r = np.sqrt(r_sq)
        th = np.arctan(r)
        thetaSq = th**2

        th_radial = np.ones(len(point_3d))
        theta2is = thetaSq.copy()

        ## radial distortion
        for i in range(numK):
            th_radial += theta2is * self.intrinsics[startK + i]
            theta2is *= thetaSq

        th_divr = th / r
        th_divr[r < eps] = 1

        xr_yr = (th_radial.reshape(-1, 1).repeat(2, 1) * th_divr.reshape(-1, 1).repeat(2, 1)) * ab
        xr_yr_squared_norm = (xr_yr**2).sum(axis=1)

        uvDistorted = xr_yr.copy()
        temp = 2.0 * xr_yr * self.intrinsics[startP:startP+2]
        uvDistorted += temp * xr_yr + xr_yr_squared_norm.reshape(-1, 1).repeat(2, 1)  * self.intrinsics[startP:startP+2]

        radialPowers2And4 = np.concatenate([xr_yr_squared_norm.reshape(-1, 1), (xr_yr_squared_norm**2).reshape(-1, 1)], axis=1)

        uvDistorted[:, 0] += (self.intrinsics[startS:startS+2] * radialPowers2And4).sum(axis=1)
        uvDistorted[:, 1] += (self.intrinsics[startS+2:] * radialPowers2And4).sum(axis=1)

        point_2d = self.intrinsics[0] * uvDistorted + self.intrinsics[1:3]

        point_2d[is_point_behind_camera] = -1

        return point_2d
