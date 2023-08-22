import numpy as np
import os
import cv2
from scipy.optimize import least_squares
import random
import pycolmap
from utils.keypoints_info import COCO_KP_ORDER
import concurrent.futures

##------------------------------------------------------------------------------------
## performs triangulation
class Triangulator: 
    def __init__(self, cfg, time_stamp, camera_names, cameras, secondary_camera_names, secondary_cameras, pose2d, humans):
        self.cfg = cfg
        self.camera_names = camera_names
        self.cameras = cameras
        self.time_stamp = time_stamp

        self.secondary_camera_names = secondary_camera_names
        self.secondary_cameras = secondary_cameras

        self.humans = humans

        self.keypoint_thres = self.cfg.POSE3D.KEYPOINTS_THRES
        self.bbox_area_thres = self.cfg.POSE3D.BBOX_AREA_THRES 
        self.n_iters = self.cfg.POSE3D.NUM_ITERS
        self.reprojection_error_epsilon = self.cfg.POSE3D.REPROJECTION_ERROR_EPSILON
        self.min_views = self.cfg.POSE3D.MIN_VIEWS 
        self.min_inliers = self.cfg.POSE3D.MIN_INLIER_VIEWS
        self.secondary_min_views = self.cfg.POSE3D.SECONDARY_MIN_VIEWS
        self.include_confidence = self.cfg.POSE3D.INCLUDE_CONFIDENCE
        
        self.coco_17_keypoints_idxs = np.array(range(17)) ## indexes of the COCO keypoints
        # self.coco_23_keypoints_idxs = np.array(range(23)) ## indexes of the COCO keypoints + 6 feet keypoints

        self.keypoints_idxs = self.coco_17_keypoints_idxs
        # self.keypoints_idxs = self.coco_23_keypoints_idxs

        self.num_keypoints = len(self.keypoints_idxs)

        ## parse the pose2d results, reaarange from camera view to human
        ## pose2d is a dictionary, 
        ## key = (camera_name, camera_mode), val = pose2d_results
        ## restructure to (human_id) = [(camera_name, camera_mode): pose2d]
        self.pose2d = {}

        for (camera_name, camera_mode), pose2d_results in pose2d.items():

            is_camera_primary = (camera_name, camera_mode) in list(self.cameras.keys())
            is_camera_secondary = (camera_name, camera_mode) in list(self.secondary_cameras.keys())

            ## skip if camera is not the choosen view for triangulation
            if is_camera_primary == False and is_camera_secondary == False:
                continue

            if is_camera_primary == True:
                choosen_camera = self.cameras[(camera_name, camera_mode)]

            else:
                choosen_camera = self.secondary_cameras[(camera_name, camera_mode)]

            image_area = choosen_camera.get_image_area()
            num_humans = len(pose2d_results) ## number of humans detected in this view

            for i in range(num_humans):
                pose2d_result = pose2d_results[i]
                human_name = pose2d_result['human_name']
                bbox = pose2d_result['bbox']

                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_ratio = bbox_area/image_area

                ## initialize if not in dict
                if human_name not in self.pose2d.keys():
                    self.pose2d[human_name] = {}  

                keypoints = pose2d_result['keypoints'][self.keypoints_idxs] ##only get the coco keypoints

                ## if it is an ego camera, we need to rotate the 2d pose to original aria frame
                if camera_name.startswith('aria'):
                    keypoints = choosen_camera.get_rotated_pose2d(pose2d=keypoints) ## go from human visual to aria format

                self.pose2d[human_name][(camera_name, camera_mode)] =  keypoints

                ### if human detection is too small relative to the image, remove from triangulation by setting confidnece to 0
                if area_ratio < self.bbox_area_thres:
                    self.pose2d[human_name][(camera_name, camera_mode)][:, 2] = 0 

                for k in range(len(self.cfg.POSE3D.OVERRIDE.TIMESTAMPS)):

                    if self.time_stamp == self.cfg.POSE3D.OVERRIDE.TIMESTAMPS[k] and \
                        human_name == self.cfg.POSE3D.OVERRIDE.HUMAN_NAMES[k] and \
                        str(camera_name) in self.cfg.POSE3D.OVERRIDE.EXO_CAMERAS[k]:

                        ignore_kps_idxs = np.array(self.cfg.POSE3D.OVERRIDE.KEYPOINT_IDXS[k])
                        self.pose2d[human_name][(camera_name, camera_mode)][ignore_kps_idxs, 2] = 0 

        return


    # ##-----------------------------------------
    def run(self, debug=False, return_error=False):
        points_3d = {}
        points_3d_error = {}

        ###-----------------------------------------
        ## proj_matricies is the extrinsics
        ## points are the rays in 3D
        for human_name in sorted(self.pose2d.keys()):
            points_3d[human_name] = np.zeros((self.num_keypoints, 4))
            points_3d_error[human_name] = np.zeros((self.num_keypoints, 1))
            error = 0

            for keypoint_idx in range(self.num_keypoints):
                proj_matricies = []
                points = []
                choosen_cameras = []

                ##------------------loop over primary views-------------------------------------
                for view_idx, (camera_name, camera_mode) in enumerate(self.pose2d[human_name].keys()):

                    ## skip if its a secondary camera
                    if (camera_name, camera_mode) not in self.camera_names:
                        continue

                    point_2d = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, :2] ## chop off the confidnece
                    confidence = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, 2]


                    ##---------use high confidence predictions------------------
                    if confidence > self.keypoint_thres:
                        extrinsics = self.cameras[(camera_name, camera_mode)].extrinsics[:3, :] ## 3x4
                        ray_3d = self.cameras[(camera_name, camera_mode)].cam_from_image(point_2d=point_2d)
                        
                        assert(len(ray_3d) == 3 and ray_3d[2] == 1)  

                        point = ray_3d.copy()
                        point[2] = confidence ## add keypoint confidence to the point
                        points.append(point)
                        proj_matricies.append(extrinsics)
                        choosen_cameras.append((camera_name, camera_mode)) ## camera choosen for triangulation for this point


                ##------------------loop over secondary views-------------------------------------
                secondary_proj_matricies = []
                secondary_points = []
                secondary_choosen_cameras = []

                for view_idx, (camera_name, camera_mode) in enumerate(self.pose2d[human_name].keys()):

                    ## skip if its a primary camera
                    if (camera_name, camera_mode) not in self.secondary_camera_names:
                        continue

                    point_2d = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, :2] ## chop off the confidnece
                    confidence = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, 2]

                    ##---------use high confidence predictions------------------
                    if confidence > self.keypoint_thres:
                        extrinsics = self.secondary_cameras[(camera_name, camera_mode)].extrinsics[:3, :] ## 3x4
                        ray_3d = self.secondary_cameras[(camera_name, camera_mode)].cam_from_image(point_2d=point_2d)
                        
                        assert(len(ray_3d) == 3 and ray_3d[2] == 1)  

                        point = ray_3d.copy()
                        point[2] = confidence ## add keypoint confidence to the point
                        secondary_points.append(point)
                        secondary_proj_matricies.append(extrinsics)
                        secondary_choosen_cameras.append((camera_name, camera_mode)) ## camera choosen for triangulation for this point

                ##---------------------------------------------------------------------------------------------
                use_secondary = False
                if len(points) >= self.min_views:
                    ## triangulate for a single point
                    point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matricies, points, \
                                n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

                    ## too few inliers from primary view
                    if len(inlier_views) < self.min_views:
                        use_secondary = True
                        point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matricies + secondary_proj_matricies, points + secondary_points, \
                                n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

                        if debug == True:
                            print('kps_idx:{} kps_name:{} kps_error:{}, inliers:{}, {}'.format(keypoint_idx, COCO_KP_ORDER[keypoint_idx], round(reprojection_error_vector.mean(), 4), \
                                len(inlier_views), [(choosen_cameras + secondary_choosen_cameras)[index][0] for index in inlier_views]))

                    if debug == True and use_secondary == False:
                        print('kps_idx:{} kps_name:{} kps_error:{}, inliers:{}, {}'.format(keypoint_idx, COCO_KP_ORDER[keypoint_idx], round(reprojection_error_vector.mean(), 4), len(inlier_views), [choosen_cameras[index][0] for index in inlier_views]))
                    error += reprojection_error_vector.mean()

                    points_3d[human_name][keypoint_idx, :3] = point_3d
                    points_3d[human_name][keypoint_idx, 3] = 1 ## mark as valid

                    points_3d_error[human_name][keypoint_idx, 0] = reprojection_error_vector.mean()

                elif len(points + secondary_points) >= self.secondary_min_views:
                    ## use primary + secondary cameras for these keypoints
                    point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matricies + secondary_proj_matricies, points + secondary_points, \
                                n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

                    if debug == True:
                        print('kps_idx:{} kps_name:{} kps_error:{}, inliers:{}, {}'.format(keypoint_idx, COCO_KP_ORDER[keypoint_idx], round(reprojection_error_vector.mean(), 4), \
                            len(inlier_views), [(choosen_cameras + secondary_choosen_cameras)[index][0] for index in inlier_views]))
                    error += reprojection_error_vector.mean()

                    points_3d[human_name][keypoint_idx, :3] = point_3d
                    points_3d[human_name][keypoint_idx, 3] = 1 ## mark as valid

                    points_3d_error[human_name][keypoint_idx, 0] = reprojection_error_vector.mean()

            if debug == True:
                print('{}, error:{}'.format(human_name, error))
        
        if return_error == True:
            return points_3d, points_3d_error

        return points_3d
    

    def process_keypoint(self, human_name, keypoint_idx, debug=False):
        proj_matrices = []
        points = []
        chosen_cameras = []

        for view_idx, (camera_name, camera_mode) in enumerate(self.pose2d[human_name].keys()):
            if (camera_name, camera_mode) not in self.camera_names:
                continue

            point_2d = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, :2]
            confidence = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, 2]

            if confidence > self.keypoint_thres:
                extrinsics = self.cameras[(camera_name, camera_mode)].extrinsics[:3, :]
                ray_3d = self.cameras[(camera_name, camera_mode)].cam_from_image(point_2d=point_2d)

                assert(len(ray_3d) == 3 and ray_3d[2] == 1) 

                point = ray_3d.copy()
                point[2] = confidence 
                points.append(point)
                proj_matrices.append(extrinsics)
                chosen_cameras.append((camera_name, camera_mode))

        secondary_proj_matrices = []
        secondary_points = []
        secondary_chosen_cameras = []

        for view_idx, (camera_name, camera_mode) in enumerate(self.pose2d[human_name].keys()):
            if (camera_name, camera_mode) not in self.secondary_camera_names:
                continue

            point_2d = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, :2]
            confidence = self.pose2d[human_name][(camera_name, camera_mode)][keypoint_idx, 2]

            if confidence > self.keypoint_thres:
                extrinsics = self.secondary_cameras[(camera_name, camera_mode)].extrinsics[:3, :]
                ray_3d = self.secondary_cameras[(camera_name, camera_mode)].cam_from_image(point_2d=point_2d)

                assert(len(ray_3d) == 3 and ray_3d[2] == 1)  

                point = ray_3d.copy()
                point[2] = confidence 
                secondary_points.append(point)
                secondary_proj_matrices.append(extrinsics)
                secondary_chosen_cameras.append((camera_name, camera_mode))

        use_secondary = False
        points_3d = np.zeros(4)
        points_3d_error = np.zeros(1)

        if len(points) >= self.min_views:
            point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matrices, points, n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

            if len(inlier_views) < self.min_views:
                use_secondary = True
                point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matrices + secondary_proj_matrices, points + secondary_points, n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

            points_3d[:3] = point_3d
            points_3d[3] = 1
            points_3d_error[0] = reprojection_error_vector.mean()

            if debug == True:
                print('kps_idx:{} kps_name:{} kps_error:{}, inliers:{}, {}'.format(keypoint_idx, COCO_KP_ORDER[keypoint_idx], reprojection_error_vector.mean(), \
                    len(inlier_views), [(chosen_cameras + secondary_chosen_cameras)[index][0] for index in inlier_views]))
                
            if debug == True and use_secondary == False:
                print('kps_idx:{} kps_name:{} kps_error:{}, inliers:{}, {}'.format(keypoint_idx, COCO_KP_ORDER[keypoint_idx], reprojection_error_vector.mean(), len(inlier_views), [chosen_cameras[index][0] for index in inlier_views]))

        elif len(points + secondary_points) >= self.secondary_min_views:
            point_3d, inlier_views, reprojection_error_vector = self.triangulate_ransac(proj_matrices + secondary_proj_matrices, points + secondary_points, n_iters=self.n_iters, reprojection_error_epsilon=self.reprojection_error_epsilon, direct_optimization=True)

            points_3d[:3] = point_3d
            points_3d[3] = 1
            points_3d_error[0] = reprojection_error_vector.mean()

        return points_3d.reshape(1, -1), points_3d_error.reshape(1, -1)


    def run_parallel(self, debug=False, return_error=False):
        points_3d = {}
        points_3d_error = {}

        for human_name in sorted(self.pose2d.keys()):
            points_3d[human_name] = np.zeros((self.num_keypoints, 4))
            points_3d_error[human_name] = np.zeros((self.num_keypoints, 1))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_keypoint = {executor.submit(self.process_keypoint, human_name, keypoint_idx, debug): keypoint_idx for keypoint_idx in range(self.num_keypoints)}
                for future in concurrent.futures.as_completed(future_to_keypoint):
                    keypoint_idx = future_to_keypoint[future]
                    try:
                        points_3d_result, points_3d_error_result = future.result()

                        points_3d[human_name][keypoint_idx] = points_3d_result[0]
                        points_3d_error[human_name][keypoint_idx] = points_3d_error_result[0]

                    except Exception as exc:
                        print(f'keypoint {keypoint_idx} generated an exception: {exc}')
            
            if debug:
                print('{} mean_error:{}'.format(human_name, points_3d_error[human_name].mean()))
                print()

        if return_error == True:
            return points_3d, points_3d_error

        return points_3d


    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/models/triangulation.py#L72
    def triangulate_ransac(self, proj_matricies, points, n_iters=50, reprojection_error_epsilon=0.1, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()


        ## create a list of all possible pairs of views
        view_pairs = []
        for i in range(n_views):
            for j in range(i+1, n_views):
                view_pairs.append([i, j])
        
        ## iterate over all possible pairs of views
        for i in range(len(view_pairs)):
            sampled_views = view_pairs[i]

        # for i in range(n_iters):
        #     sampled_views = sorted(random.sample(view_set, 2)) ## sample two views

            keypoint_3d_in_base_camera = self.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]

                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        ##-------------------------------
        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = self.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points, self.include_confidence)
        reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list, reprojection_error_vector

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L113
    def triangulate_point_from_multiple_views_linear(self, proj_matricies, points, include_confidence=True):
        """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
        For more information look at "Multiple view geometry in computer vision",
        Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
        Args:
            proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
            points numpy array of shape (N, 3): sequence of points' coordinates and confidence
        Returns:
            point_3d numpy array of shape (3,): triangulated point
        """
        assert len(proj_matricies) == len(points)

        points_confidence = points[:, 2].copy()
        points = points[:, :2].copy()

        ###-----normalize points_confidence-----
        points_confidence /= points_confidence.max()

        n_views = len(proj_matricies)
        A = np.zeros((2 * n_views, 4))
        for j in range(len(proj_matricies)):
            A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
            A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

            ## weight by the point confidence
            if include_confidence == True:
                A[j * 2 + 0] *= points_confidence[j]
                A[j * 2 + 1] *= points_confidence[j]

        u, s, vh =  np.linalg.svd(A, full_matrices=False)
        point_3d_homo = vh[3, :]

        point_3d = self.homogeneous_to_euclidean(point_3d_homo)

        return point_3d

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L186
    def calc_reprojection_error_matrix(self, keypoints_3d, keypoints_2d_list, proj_matricies):
        reprojection_error_matrix = []
        for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):

            if len(keypoints_2d) == 3:
                keypoints_2d = keypoints_2d[:2] ## chop off the confidence

            keypoints_2d_projected = self.project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
            reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
            reprojection_error_matrix.append(reprojection_error)

        return np.vstack(reprojection_error_matrix).T

    ##-----------------------------------------
    def project_3d_points_to_image_plane_without_distortion(self, proj_matrix, points_3d, convert_back_to_euclidean=True):
        """Project 3D points to image plane not taking into account distortion
        Args:
            proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
            points_3d numpy array or torch tensor of shape (N, 3): 3D points
            convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                            NOTE: division by zero can be here if z = 0
        Returns:
            numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
        """
        if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
            result = self.euclidean_to_homogeneous(points_3d) @ proj_matrix.T
            if convert_back_to_euclidean:
                result = self.homogeneous_to_euclidean(result)
            return result
        elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
            result = self.euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
            if convert_back_to_euclidean:
                result = self.homogeneous_to_euclidean(result)
            return result
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L72
    def homogeneous_to_euclidean(self, points):
        """Converts homogeneous points to euclidean
        Args:
            points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
        Returns:
            numpy array or torch tensor of shape (N, M): euclidean points
        """
        if isinstance(points, np.ndarray):
            return (points.T[:-1] / points.T[-1]).T
        elif torch.is_tensor(points):
            return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L55
    def euclidean_to_homogeneous(self, points):
        """Converts euclidean points to homogeneous
        Args:
            points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
        Returns:
            numpy array or torch tensor of shape (N, M + 1): homogeneous points
        """
        if isinstance(points, np.ndarray):
            return np.hstack([points, np.ones((len(points), 1))])
        elif torch.is_tensor(points):
            return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")