import numpy as np
import os
import cv2
import trimesh
import pycolmap
from utils.transforms import linear_transform
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

##------------------------------------------------------------------------------------
class ExoCamera:
    def __init__(self, cfg, root_dir, colmap_dir, exo_camera_name='cam01', coordinate_transform=None, reconstruction=None, exo_camera_mapping=None, max_time_stamps=4):
        self.cfg = cfg
        self.root_dir = root_dir
        self.exo_camera_name = exo_camera_name
        self.camera_name = exo_camera_name

        self.exo_camera_id = int(self.exo_camera_name.replace('cam', ''))
        self.colmap_dir = colmap_dir
        self.type_string = 'rgb'
        self.camera_type = 'exo'

        self.images_path = os.path.join(self.root_dir, self.exo_camera_name, 'images')
        self.intrinsics_calibration_file = os.path.join(self.colmap_dir, 'cameras.txt')
        self.extrinsics_calibration_file = os.path.join(self.colmap_dir, 'images.txt')

        self.coordinate_transform = coordinate_transform
        self.reconstruction = reconstruction

        ###----------set image width and height----------
        self.image_height, self.image_width = self.set_image_resolution()

        ##--------load intrinsics------------
        ### load intrisncis for other cameras if colmap did not converge for these gopros
        ## intrisicnsi of gopors are approximately similar for same resolution
        if self.exo_camera_name in self.cfg.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS:
            idx = self.cfg.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS.index(self.exo_camera_name)
            other_exo_camera_name = self.cfg.CALIBRATION.MANUAL_INTRINSICS_FROM_EXO_CAMERAS[idx]
            self.colmap_camera_id = exo_camera_mapping[other_exo_camera_name]
            self.intrinsics = self.reconstruction.cameras[self.colmap_camera_id]

        else:
            self.colmap_camera_id = exo_camera_mapping[self.exo_camera_name]
            self.intrinsics = self.reconstruction.cameras[self.colmap_camera_id]

        ##------------load all extrinsics-----
        self.all_extrinsics = {}
        self.all_extrinsics_image = {} ## image is pycolmap object

        ###-------if manual calibration-------
        if self.exo_camera_name in self.cfg.CALIBRATION.MANUAL_EXO_CAMERAS:
            extrinsics = np.load(os.path.join(self.colmap_dir, '{}.npy'.format(self.exo_camera_name)))
            self.all_extrinsics[1] = extrinsics[:3, :]
            self.all_extrinsics_image[1] = None ## never used

        else:

            for image_id, image in self.reconstruction.images.items():
                image_path = image.name
                image_camera_name = image_path.split('/')[0]
                time_stamp = int((image_path.split('/')[1]).replace('.jpg', ''))

                if image_camera_name == self.exo_camera_name:
                    self.all_extrinsics[time_stamp] = image.projection_matrix() ## 3 x 4
                    self.all_extrinsics_image[time_stamp] = image

                if len(self.all_extrinsics.keys()) > max_time_stamps:
                    break
            
        ##------------vis--------------
        self.alpha = 0.7
        self.color = [255, 255, 0] ## bgr

        return  

    ##--------------------------------------------------------
    def set_image_resolution(self):
        image_path = self.get_image_path(time_stamp=1)
        image = cv2.imread(image_path)
        image_height = image.shape[0]
        image_width = image.shape[1]
        return image_height, image_width

    ##--------------------------------------------------------
    def set_closest_calibration(self, time_stamp):
        min_dist_time_stamp = None
        min_dist = None

        ## nearest neighbour by time stamps
        for calib_time_stamp in self.all_extrinsics.keys():
            dist = abs(calib_time_stamp - time_stamp)

            if min_dist == None or dist < min_dist:
                min_dist = dist
                min_dist_time_stamp = calib_time_stamp

        if min_dist_time_stamp == None:
            print('{} extrinsics not found, you should be in manual calibration mode or better know what you are doing! returning dummy_extrinsics'.format(self.exo_camera_name))
            dummy_extrinsics = (np.eye(4))[:3, :]
            return None, dummy_extrinsics

        self.calib_time_stamp = min_dist_time_stamp
        return self.all_extrinsics_image[min_dist_time_stamp], self.all_extrinsics[min_dist_time_stamp]

    ##--------------------------------------------------------
    def update(self, time_stamp):
        self.extrinsics_image, self.extrinsics = self.set_closest_calibration(time_stamp=time_stamp)
        self.raw_extrinsics = np.concatenate([self.extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        self.extrinsics = np.dot(self.raw_extrinsics, self.coordinate_transform)
        self.time_stamp = time_stamp

        self.location = self.get_location()

        # ## frustum rays of the camera in camera coordinate system
        # ray1 = self.cam_from_image(np.array([0, 0]))
        # ray2 = self.cam_from_image(np.array([self.image_width - 1, 0]))
        # ray3 = self.cam_from_image(np.array([0, self.image_height - 1]))
        # ray4 = self.cam_from_image(np.array([self.image_width - 1, self.image_height - 1]))        
        # ray5 = self.cam_from_image(np.array([self.image_width/2, self.image_height/2]))        
        
        # # ## normalize
        # ray1 = ray1/np.sqrt((ray1**2).sum())
        # ray2 = ray2/np.sqrt((ray2**2).sum())
        # ray3 = ray3/np.sqrt((ray3**2).sum())
        # ray4 = ray4/np.sqrt((ray4**2).sum())
        # ray5 = ray5/np.sqrt((ray5**2).sum())

        # n_vec = ray5
        # p1 = ray1/np.dot(ray1, n_vec) - n_vec
        # p2 = ray2/np.dot(ray2, n_vec) - n_vec
        # p3 = ray3/np.dot(ray3, n_vec) - n_vec
        # p4 = ray4/np.dot(ray4, n_vec) - n_vec
        # self.frustum_rays = [ray1, ray2, ray3, ray4, ray5]
        # self.frustum_points = [p1, p2, p3, p4]
        # self.frustum_polygon = Polygon([p1, p2, p3, p4])

        return

    ##--------------------------------------------------------
    def init_undistort_map(self):
        ##--------------load undistorted K-------------------------------
        params = self.intrinsics.params
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        k1 = params[4]
        k2 = params[5]
        k3 = params[6]
        k4 = params[7]

        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.D_fisheye = np.array([k1, k2, k3, k4,])
        self.K_undistorted = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D_fisheye, (self.image_width, self.image_height), np.eye(3), balance=1.0)
        return

    def get_undistorted_image(self, time_stamp):
        image = self.get_image(time_stamp)
        undistorted_image = cv2.fisheye.undistortImage(image, self.K, self.D_fisheye, None, self.K)
        return undistorted_image

    def read_undistorted_image(self, time_stamp):
        image_path = os.path.join(self.images_path.replace('/images', '/undistorted_images'), '{:05d}.jpg'.format(time_stamp))
        image = cv2.imread(image_path)
        return image

    ##--------------------------------------------------------
    def get_image_path(self, time_stamp):
        image_path = os.path.join(self.images_path, '{:05d}.jpg'.format(time_stamp))
        return image_path

    def get_image(self, time_stamp):
        image_path = self.get_image_path(time_stamp=time_stamp) 
        image = cv2.imread(image_path)
        return image

    def get_image_area(self):
        return self.image_width * self.image_height

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


    ## point_3d in global coordinate system
    def check_point_behind_camera_3d(self, point_3d):
        point_3d_prime = (linear_transform(points_3d=np.expand_dims(point_3d, axis=0), T=self.coordinate_transform))[0]
        camera_location_prime = (linear_transform(points_3d=np.expand_dims(self.location, axis=0), T=self.coordinate_transform))[0] ## remove fake batch

        vec_to_point = point_3d_prime - camera_location_prime ## vector in the colmap coordinate system
        is_behind_vec = np.dot(self.extrinsics_image.viewing_direction(), vec_to_point)

        return is_behind_vec < 0

    ## to camera coordinates 3D from world cooredinate 3D
    def cam_from_world(self, point_3d):
        assert(point_3d.shape[0] == 3)
        point_3d_cam = (linear_transform(points_3d=np.expand_dims(point_3d, axis=0), T=self.extrinsics))[0]

        ## Note, this function is commented for now. self.location bug with scaling to COLMAP -> ARia is the root cause.
        ## till then using this function is not reliable!
        # if self.check_point_behind_camera_3d(point_3d):
        #     point_3d_cam = np.array([point_3d_cam[0], point_3d_cam[1], -1]) ## purposeful -1 which will trigget check_point_behind_camera check later

        return point_3d_cam

    # https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    def check_point_outside_frustrum(self, point_3d):
        p_vec = point_3d/np.sqrt((point_3d**2).sum())
        n_vec = self.frustum_rays[-1] # the last ray, corresponding to the center of the image

        p_prime_vec = p_vec/np.dot(p_vec, n_vec) - n_vec
        p1 = self.frustum_points[0]
        p2 = self.frustum_points[1]
        p3 = self.frustum_points[2]
        p4 = self.frustum_points[3]

        return False

    ## reference: https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    def check_point_behind_camera(self, point_3d):
        assert(point_3d.shape[0] == 3)
        return True if point_3d[2] < 0 else False

    def image_from_cam(self, point_3d, eps=1e-9):

        if self.check_point_behind_camera(point_3d):
            return np.array([-1, -1]) ## out of bound

        # # ##---convert to (u, v, 1) format as done in pycolmap
        point_3d = point_3d/point_3d[2] ## (u, v, 1) normalization

        point_2d = self.intrinsics.world_to_image(point_3d[:2])    
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
        point_3d_world = (linear_transform(points_3d=np.expand_dims(point_3d, axis=0), T=self.inv_extrinsics))[0]
        return point_3d_world

    ## to camera coordinates 3D from image cooredinate 2D
    ## reference: https://www.internalfb.com/code/fbsource/[0c9e159d90aee8bfeff67a6e2066726a5ecc1796]/arvr/projects/surreal/experiments/PseudoGT/increAssoRecon/core/geometry.cpp?lines=753
    def cam_from_image(self, point_2d):
        point_3d = np.ones(3) ## initialize
        point_3d[0] = 0; point_3d[1] = 0; point_3d[2] = 1 ## initialize

        ray_3d = self.intrinsics.image_to_world(point_2d)
        point_3d[0] = ray_3d[0]
        point_3d[1] = ray_3d[1]

        return point_3d

    ##--------------------------------------------------------
    ## return 3 x 3 rotation, world to cam
    def get_rotation(self):
        return self.extrinsics[:3, :3]

    ## return 3 x 1 translation, world to cam
    def get_translation(self):
        return self.extrinsics[:3, 3]

    def get_location(self):
        # rotmat = self.get_rotation()
        # translation = self.get_translation()
        # location = -1*np.dot(rotmat.T, translation)
        self.inv_extrinsics = np.linalg.inv(self.extrinsics)
        center = np.dot(self.inv_extrinsics, np.array([0, 0, 0, 1]).reshape(4, 1))
        location = center[:3].reshape(-1)

        return location

    ##------------------------bbox-------------------------
    ## this works for VRS format true
    def get_bbox_2d(self, aria_human):
        min_vertices = self.cfg.BBOX.MIN_VERTICES
        bbox_thres = self.cfg.BBOX.EXO.MIN_AREA_RATIO
        max_aspect_ratio_thres = self.cfg.BBOX.EXO.MAX_ASPECT_RATIO
        min_aspect_ratio_thres = self.cfg.BBOX.EXO.MIN_ASPECT_RATIO

        bbox_3d = aria_human.get_bbox_3d()
        bbox_2d_all = self.project_batch(batch_point_3d=bbox_3d)
        bbox_2d_all = self.check_bounds(bbox_2d_all)

        if len(bbox_2d_all) < min_vertices:
            bbox_3d = aria_human.get_better_bbox_3d(num_points=800)
            bbox_2d_all = self.project_batch(batch_point_3d=bbox_3d)
            bbox_2d_all = self.check_bounds(bbox_2d_all)

        ## the human is out of sight!
        if len(bbox_2d_all) == 0:
            return None       

        x1 = bbox_2d_all[:, 0].min()
        x2 = bbox_2d_all[:, 0].max()
        y1 = bbox_2d_all[:, 1].min()
        y2 = bbox_2d_all[:, 1].max()

        bbox_width = (x2 - x1)
        bbox_height = (y2 - y1)
        area = bbox_width * bbox_height

        image_area = self.get_image_area()
        bbox_area_ratio = area*1.0/image_area

        ## if bbox is too small
        if bbox_area_ratio < bbox_thres:
            return None

        aspect_ratio = bbox_height/bbox_width ## height/width

        ## the bbox is too skewed, height is large compared to width
        if aspect_ratio > max_aspect_ratio_thres or aspect_ratio < min_aspect_ratio_thres:
            return None

        bbox_2d = np.array([round(x1), round(y1), round(x2), round(y2)]) ## convert to integers

        return bbox_2d

     ##------------------------bbox just for the head-------------------------
    ## this works for VRS format true
    def get_head_bbox_2d(self, aria_human):
        bbox_3d = aria_human.get_head_bbox_3d() ## sphere of the bbox of the head
        bbox_2d_all = self.project_batch(batch_point_3d=bbox_3d)
        bbox_2d_all = self.check_bounds(bbox_2d_all)

        ## the human is out of sight!
        if len(bbox_2d_all) == 0:
            return None       

        x1 = bbox_2d_all[:, 0].min()
        x2 = bbox_2d_all[:, 0].max()
        y1 = bbox_2d_all[:, 1].min()
        y2 = bbox_2d_all[:, 1].max()

        bbox_2d = np.array([round(x1), round(y1), round(x2), round(y2)]) ## convert to integers

        return bbox_2d

    ## check for points inside the image
    def check_bounds(self, bbox_2d_all):
        is_valid = (bbox_2d_all[:, 0] >= 0) * (bbox_2d_all[:, 0] <= self.image_width) * \
                    (bbox_2d_all[:, 1] >= 0) * (bbox_2d_all[:, 1] <= self.image_height)
        return bbox_2d_all[is_valid]

    ##------------------------------------------------
    # https://github.com/colmap/colmap/blob/d6f528ab59fd653966e857f8d0c2203212563631/scripts/python/read_write_model.py#L453
    def qvec2rotmat(self, qvec):
      return np.array([
          [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
           2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
           2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
          [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
           1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
           2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
          [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
           2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
           1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

    def rotmat2qvec(self, R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    ##------------------------------------------------
    def get_sphere_mesh(self, point_3d, radius=0.2):
        transform = np.eye(4) ##4x4
        transform[:3, 3] = point_3d ## place the sphere at the location
        mesh = trimesh.primitives.Sphere(radius=radius)
        mesh.apply_transform(transform)
        mesh.visual.face_colors = [self.color[2], self.color[1], self.color[0], 255*self.alpha] ## note the colors are bgr
        return mesh

    ##------------get the location of the aria in 3d-----------
    def get_aria_location(self, point_3d):
        point_2d = self.project(point_3d)
        is_valid = self.is_inside_bound(point_2d)

        return point_2d, is_valid

    def is_inside_bound(self, point_2d):
        is_valid = (point_2d[0] >= 0) * (point_2d[0] < self.image_width) *\
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

    # ##--------------------------------------------------------
    # ## to camera coordinates 3D from world cooredinate 3D
    def vec_cam_from_world(self, point_3d):
        point_3d_cam = linear_transform(point_3d, self.extrinsics)
        return point_3d_cam

    # ## reference: https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    # def vec_check_point_behind_camera(self, point_3d):    
    #     return point_3d[:, 2] < 0 


    # # to image coordinates 2D from camera cooredinate 3D
    # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    def vec_image_from_cam(self, point_3d, eps=1e-9):
        fx, fy, cx, cy, k1, k2, k3, k4 = self.intrinsics.params

        x = point_3d[:, 0]
        y = point_3d[:, 1]
        z = point_3d[:, 2]

        a = x/z; b = y/z
        r = np.sqrt(a*a + b*b)
        theta = np.arctan(r)

        theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
        x_prime = (theta_d/r)*a
        y_prime = (theta_d/r)*b

        u = fx*(x_prime + 0) + cx
        v = fy*y_prime + cy

        point_2d = np.concatenate([u.reshape(-1, 1), v.reshape(-1, 1)], axis=1)

        return point_2d