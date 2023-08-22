import numpy as np
import os
import cv2
from .aria_camera import AriaCamera
import trimesh
import pickle
from utils.keypoints_info import COCO_KP_ORDER
from utils.transforms import distance_from_plane, projected_point_to_plane, is_point_on_plane, plane_unit_normal, get_point_on_plane
from utils.refine_pose3d import refine_pose3d
from models.fit_pose3d import fit_pose3d

##------------------------------------------------------------------------------------
class AriaHuman:
    def __init__(self, cfg, root_dir, human_name, human_id=0, ground_plane=None, coordinate_transform=None, color=[255, 0, 0], alpha=0.7):
        self.cfg = cfg
        self.root_dir = root_dir
        self.human_name = human_name
        self.human_id = human_id
        self.ground_plane = ground_plane
        self.unit_normal = plane_unit_normal(self.ground_plane)
        self.point_on_plane = get_point_on_plane(self.ground_plane)
        
        self.calibration_path = os.path.join(self.root_dir, self.human_name, 'calib')
        self.total_time = len(sorted(os.listdir(self.calibration_path))) if self.cfg.SEQUENCE_TOTAL_TIME == -1 else self.cfg.SEQUENCE_TOTAL_TIME
        assert(self.total_time <= len(sorted(os.listdir(self.calibration_path))))

        self.images_path = os.path.join(self.root_dir, self.human_name, 'images')

        ## we go from global colmap to aria coordinate
        ## then the self.extrinsics is valid, world to cam cooredinates of the aria 
        self.coordinate_transform = coordinate_transform

        ###--------------------------------------------------------
        self.rgb_cam = AriaCamera(cfg=cfg, human_id=self.human_id, camera_name=human_name, type_string='rgb', calibration_path=self.calibration_path, images_path=self.images_path) ## 0
        self.left_cam = AriaCamera(cfg=cfg, human_id=self.human_id, camera_name=human_name, type_string='left', calibration_path=self.calibration_path, images_path=self.images_path) ## 1
        self.right_cam = AriaCamera(cfg=cfg, human_id=self.human_id, camera_name=human_name, type_string='right', calibration_path=self.calibration_path, images_path=self.images_path) ## 2

        self.location = None ##in the global frame
        self.time_stamp = None

        self.color = color
        if self.human_id == 0:
            # self.color = [255, 0, 0] ## bgr, blue
            self.color = [204, 0, 0] ## bgr, blue
            self.color_string = 'blue'

        elif self.human_id == 1:
            # self.color = [0, 255, 0] ## bgr, green
            self.color = [0, 204, 0] ## bgr, green
            self.color_string = 'green'

        elif self.human_id == 2:
            self.color = [0, 0, 204] ## bgr, red
            self.color_string = 'red'

        elif self.human_id == 3:
            self.color = [52, 219, 235] ## bgr, yellow
            self.color_string = 'goldenrod' ## matplotlib colors

        elif self.human_id == 4:
            # self.color = [219, 112, 147] ## bgr, purple
            self.color = [204, 0, 102] ## bgr, purple
            self.color_string = 'purple' ## matplotlib colors

        elif self.human_id == 5:
            # self.color = [0, 165, 235] ## bgr, orange
            self.color = [51, 153, 255] ## bgr, orange
            self.color_string = 'orange' ## matplotlib colors

        elif self.human_id == 6:
            # self.color = [186, 152, 13] ## bgr, blue-green
            self.color = [204, 204, 0] ## bgr, blue-green
            self.color_string = 'cyan' ## matplotlib colors

        elif self.human_id == 7:
            self.color = [144, 238, 144] ## bgr, yellow
            self.color_string = 'lightgreen' ## matplotlib colors

        self.alpha = alpha

        ##----------------3d pose---------------------
        self.pose3d = None ## per time stamp
        self.pose3d_trajectory = None ## all time stamps

        return  
    
    ##--------------------------------------------------------
    def read_calibration(self, time_stamp):
        time_stamp_string = '{:05d}'.format(time_stamp)
        calibration_file = os.path.join(self.calibration_path, '{}.txt'.format(time_stamp_string))

        with open(calibration_file) as f:
            lines = f.readlines()
            lines = lines[1:] ## drop the header, eg. Serial, intrinsics (radtanthinprsim), extrinsic (3x4)
            lines = [line.strip() for line in lines]

        output = {}
        assert(len(lines) % 7 == 0) # 1 for person id, 2 lines each for rgb, left and right cams. Total 7 lines per person
        num_persons = len(lines)//7
        assert(num_persons == 1) ## we assume only single person per calib directory

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

            output[idx] = {'rgb': rgb_cam, 'left': left_cam, 'right':right_cam, 'person_id_string': person_id}

        return output[0] 

    ##--------------------------------------------------------
    def update(self, time_stamp):
        self.time_stamp = time_stamp
        calibration = self.read_calibration(time_stamp)

        rgb_intrinsics = calibration['rgb']['intrinsics']
        rgb_extrinsics = calibration['rgb']['extrinsics'] ## this is world to camera
        rgb_extrinsics = np.concatenate([rgb_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        rgb_extrinsics = np.dot(rgb_extrinsics, self.coordinate_transform) ## align the world
        self.rgb_cam.update(intrinsics=rgb_intrinsics, extrinsics=rgb_extrinsics)

        left_intrinsics = calibration['left']['intrinsics']
        left_extrinsics = calibration['left']['extrinsics']
        left_extrinsics = np.concatenate([left_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        left_extrinsics = np.dot(left_extrinsics, self.coordinate_transform) ## align the world
        self.left_cam.update(intrinsics=left_intrinsics, extrinsics=left_extrinsics)

        right_intrinsics = calibration['right']['intrinsics']
        right_extrinsics = calibration['right']['extrinsics']
        right_extrinsics = np.concatenate([right_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        right_extrinsics = np.dot(right_extrinsics, self.coordinate_transform) ## align the world
        self.right_cam.update(intrinsics=right_intrinsics, extrinsics=right_extrinsics)

        ## update location in world frame
        self.location = (self.left_cam.get_location() + self.right_cam.get_location())/2

        return

    ##--------------------------------------------------------
    def get_bbox_3d(self):
        # mesh = self.get_sphere_mesh(point_3d=self.location)
        mesh = self.get_cylinder_mesh(point_3d=self.location)
        bbox_3d = mesh.vertices ## slower but smoother
        # bbox_3d = mesh.bounding_box_oriented.vertices
        return bbox_3d
    
    ##--------------------------------------------------------
    def get_head_bbox_3d(self):
        mesh = self.get_sphere_mesh(point_3d=self.location)
        bbox_3d = mesh.vertices ## slower but smoother
        return bbox_3d

    ##--------------------------------------------------------
    def get_better_bbox_3d(self, num_points=800):
        # mesh = self.get_sphere_mesh(point_3d=self.location)
        mesh = self.get_cylinder_mesh(point_3d=self.location)

        ## randomly sample lots of points inside the mesh
        points = trimesh.sample.volume_mesh(mesh, num_points)
        return points

    ##--------------------------------------------------------
    ## returns a sphere mesh vertices centered at the location of the human head
    def get_capsule_bbox_3d(self):
        mesh = self.get_capsule_mesh(point_3d=self.location)
        bbox_3d = mesh.vertices ## slower but smoother
        return bbox_3d

    ##--------------------------------------------------------
    ## returns a sphere mesh vertices centered at the location of the human head
    def get_sphere_mesh(self, point_3d, radius=0.1):
        transform = np.eye(4) ##4x4
        transform[:3, 3] = point_3d ## place the sphere at the location
        mesh = trimesh.primitives.Sphere(radius=radius)
        mesh.apply_transform(transform)
        mesh.visual.face_colors = [self.color[2], self.color[1], self.color[0], 255*self.alpha] ## note the colors are bgr
        return mesh

    ##--------------------------------------------------------
    ### y axis in world coordinates is the gravity direction
    ### z is parallel to the ground plane
    def get_cylinder_mesh(self, point_3d, padding=1.1):
        radius = self.cfg.BBOX.ROI_CYLINDER_RADIUS

        point_3d_ = point_3d.copy() ## the location of the aria glasses
        transform = trimesh.transformations.rotation_matrix(np.deg2rad(180), [0, 1, 0]) ## angle and axis

        ## compute distance of the aria glass from the ground plane
        projected_point, distance_to_ground = projected_point_to_plane(point_3d_, self.ground_plane)

        ## debug
        # print('t:{} unit normal:{}, equation:{}'.format(self.time_stamp, self.unit_normal, self.ground_plane.get_equation()))

        ## compute the height of the human
        if self.cfg.BBOX.HUMAN_HEIGHT is None:
            human_height = padding*distance_to_ground

            point_head = projected_point + human_height*self.unit_normal
            cylinder_center = (point_head + projected_point)/2
        else:
            ## for ego 4D, glass is on the head
            human_height = self.cfg.BBOX.HUMAN_HEIGHT
            
            if np.dot(self.unit_normal, point_3d_ - self.point_on_plane) < 0:
                ## if the point is below the ground plane, then the height is negative
                human_height = -human_height

            cylinder_center = point_3d - human_height*self.unit_normal*0.5

            # print(cylinder_center, unit_normal, self.ground_plane.get_equation()) ## unit normal is changing
        
        transform[:3, 3] = cylinder_center ## place the cylinder at the hip of the human

        mesh = trimesh.primitives.Cylinder(radius=radius, height=human_height)

        mesh.apply_transform(transform)
        mesh.visual.face_colors = [self.color[2], self.color[1], self.color[0], 255*self.alpha]
        return mesh

    ##--------------------------------------------------------
    def get_capsule_mesh(self, point_3d, radius=0.3, padding=1.8):
        point_3d_ = point_3d.copy() ## the location of the aria glasses
        transform = trimesh.transformations.rotation_matrix(np.deg2rad(180), [0, 1, 0]) ## angle and axis

        ## compute distance of the aria glass from the ground plane
        projected_point, distance_to_ground = projected_point_to_plane(point_3d_, self.ground_plane)
        unit_normal = plane_unit_normal(self.ground_plane)

        human_height = padding*distance_to_ground

        point_head = projected_point + human_height*unit_normal
        cylinder_center = (point_head + projected_point)/2
        transform[:3, 3] = cylinder_center ## place the cylinder at the hip of the human

        mesh = trimesh.primitives.Capsule(radius=radius, height=human_height)

        mesh.apply_transform(transform)
        mesh.visual.face_colors = [self.color[2], self.color[1], self.color[0], 255*self.alpha]
        return mesh


    ##--------------------------------------------------------
    def set_pose3d(self, pose3d):
        self.pose3d = pose3d
        return

    ##--------------------------------------------------------
    def set_poses3d_trajectory(self, poses3d_trajectory):
        self.poses3d_trajectory = poses3d_trajectory
        return

    ##--------------------------------------------------------
    def set_smpl(self, smpl):
        self.smpl = smpl
        return

    ##--------------------------------------------------------
    def get_refine_poses3d(self):
        self.refine_poses3d_trajectory = refine_pose3d(cfg=self.cfg, human_name=self.human_name, poses=self.poses3d_trajectory) ## T x 17 x 4
        return self.refine_poses3d_trajectory    

    ##--------------------------------------------------------
    def fit_poses3d(self):
        self.fitted_poses3d_trajectory = fit_pose3d(cfg=self.cfg, human_name=self.human_name, poses_numpy=self.poses3d_trajectory) ## T x 17 x 4
        return self.fitted_poses3d_trajectory    