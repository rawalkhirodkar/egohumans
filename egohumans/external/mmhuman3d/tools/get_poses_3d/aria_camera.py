import numpy as np
import os
import cv2

##------------------------------------------------------------------------------------
def read_calibration(calibration_file):
    with open(calibration_file) as f:
        lines = f.readlines()
        lines = lines[1:] ## drop the header, eg. Serial, intrinsics (radtanthinprsim), extrinsic (3x4)
        lines = [line.strip() for line in lines]

    output = {}
    assert(len(lines) % 7 == 0) # 1 for person id, 2 lines each for rgb, left and right cams. Total 7 lines per person
    num_persons = len(lines)//7

    for idx in range(num_persons):
        data = lines[idx*7:(idx+1)*7]

        person_id = data[0]
        rgb_intrinsics = np.asarray([float(x) for x in data[1].split(' ')])
        rgb_extrinsics = np.asarray([float(x) for x in data[2].split(' ')]).reshape(4, 3).T

        left_intrinsics = np.asarray([float(x) for x in data[3].split(' ')])
        left_extrinsics = np.asarray([float(x) for x in data[4].split(' ')]).reshape(4, 3).T

        right_intrinsics = np.asarray([float(x) for x in data[5].split(' ')])
        right_extrinsics = np.asarray([float(x) for x in data[6].split(' ')]).reshape(4, 3).T

        ###--------------store everything as nested dicts---------------------
        rgb_cam = {'intrinsics': rgb_intrinsics, 'extrinsics': rgb_extrinsics}
        left_cam = {'intrinsics': left_intrinsics, 'extrinsics': left_extrinsics}
        right_cam = {'intrinsics': right_intrinsics, 'extrinsics': right_extrinsics}

        output[person_id] = {'rgb': rgb_cam, 'left': left_cam, 'right':right_cam}

    return output

##------------------------------------------------------------------------------------
class AriaCamera:
    def __init__(self, person_id='', type='rgb', intrinsics=None, extrinsics=None, timestamp=0):
        self.person_id = person_id
        self.type = type

        # the parameter vector has the following interpretation:
        # intrinsic = [f c_u c_v [k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
        self.intrinsics = intrinsics
        assert(self.intrinsics.shape[0] == 15)

        self.extrinsics = extrinsics
        # self.extrinsics = np.eye(4) ## for debug
        # print('extrinsics is identity for debug')

        assert(self.extrinsics.shape[0] == 4 and self.extrinsics.shape[1] == 4)
        self.inv_extrinsics = np.linalg.inv(self.extrinsics)

        return  

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

    ## to image coordinates 2D from camera cooredinate 3D
    ## reference: https://ghe.oculus-rep.com/minhpvo/NikosInternship/blob/04e672190a11c76d0b810ebc121f46a2aa5b67e4/aria_hand_obj/aria_utils/geometry.py#L29
    ## reference: https://www.internalfb.com/code/fbsource/[0c9e159d90aee8bfeff67a6e2066726a5ecc1796]/arvr/projects/surreal/experiments/PseudoGT/increAssoRecon/core/geometry.cpp?lines=567
    def image_from_cam(self, point_3d, eps=1e-9):
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
            return np.asarray([0, 0, 1]) ## TODO: UnitZ()? check correctness

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
    ## return 3 x 3 rotation
    def get_rotation(self):
        return self.extrinsics[:3, :3]

    ## return 3 x 1 translation
    def get_translation(self):
        return self.extrinsics[:3, 3]


















