import numpy as np
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import onnxruntime

##------------------------------------------------------------------------------------   
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

##------------------------------------------------------------------------------------
class SegmentationModel:
    def __init__(self, cfg, model_type=None, checkpoint=None, onnx_checkpoint=None):
        self.cfg = cfg
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.onnx_checkpoint = onnx_checkpoint

        self.load_model()

        self.coco_17_keypoints_idxs = np.array(range(17)) ## indexes of the COCO keypoints
        self.coco_17_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

        return 

    def load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)
        print('loading segmentation {} model from {}'.format(self.model_type, self.checkpoint))

        self.ort_session = onnxruntime.InferenceSession(self.onnx_checkpoint, 
                                                        providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'])

        return


    ####--------------------------------------------------------
    def get_segmentation(self, image_name, poses2d, debug=False):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image = image
        self.predictor.set_image(image)
        self.image_embedding = self.predictor.get_image_embedding().cpu().numpy()

        # image_name='/media/rawalk/disk1/rawalk/datasets/ego_exo/main/14_grappling/001_grappling/exo/cam01/images/00069.jpg'
        camera_name = image_name.split('/')[-3]

        segmentations = {}
        
        for human_name, pose2d in poses2d.items():

            ## get tight bbox to pose2d
            bbox = self.get_tight_bbox(pose2d) ## xyxy is the default format for the code
            segmentation = self.forward_segmentation(poses2d, bbox, human_name, debug=debug)
            segmentations[human_name] = {'segmentation': segmentation, 'kf_pose2d': pose2d, 'bbox': bbox}

            if debug:
                ## visualize segmentation, segmentation is h x w, convert to h x w x 3, write to disk
                vis_segmentation = segmentation.reshape(segmentation.shape[0], segmentation.shape[1], 1)
                vis_segmentation = np.repeat(vis_segmentation, 3, axis=2)
                vis_segmentation = vis_segmentation.astype(np.uint8)
                vis_segmentation = vis_segmentation * 255

                save_path = 'final_'+human_name + '.png'
                cv2.imwrite(save_path, vis_segmentation)

                if camera_name == 'cam08':
                    import pdb; pdb.set_trace()
                    temp = 1

        return segmentations
    
    def forward_segmentation(self, poses2d, bbox, human_name, debug=False):
        raw_pose2d = poses2d[human_name] ## 17 x 3
        pose2d = raw_pose2d[:, :2] ## 17 x 2

        positive_points = pose2d
        negative_points = [] ## all other human keypoints

        for other_human_name, raw_other_pose2d in poses2d.items():
            if other_human_name == human_name:
                continue
                
            oks = self.oks(raw_pose2d, raw_other_pose2d, keypoint_thres=0.3)

            ## check how many raw_other_pose2d keypoints are within bbox
            is_valid = (raw_other_pose2d[:, 0] > bbox[0]) * (raw_other_pose2d[:, 0] < bbox[2]) * (raw_other_pose2d[:, 1] > bbox[1]) * (raw_other_pose2d[:, 1] < bbox[3])
            
            if oks > 0.1 or is_valid.sum() > 3:
                negative_points.append(raw_other_pose2d[:, :2])
        
        # to numpy
        if len(negative_points) > 0:
            negative_points = np.concatenate(negative_points, axis=0)

        ## first infer a low resolution segmentation using positive_points only
        ## then infer a high resolution segmentation using positive_points and negative_points
        segmentation = self.forward(positive_points, negative_points)

        if debug:
            ## draw the positive and negative points on the image as blue and red circles
            ## use cv2.circle
            temp_image = self.image.copy()
            for i in range(len(positive_points)):
                cv2.circle(temp_image, (int(positive_points[i, 0]), int(positive_points[i, 1])), 3, (0, 0, 255), -1) ## blue for rgb

            for i in range(len(negative_points)):
                cv2.circle(temp_image, (int(negative_points[i, 0]), int(negative_points[i, 1])), 3, (255, 0, 0), -1)

            ## draw the segmentation on the image as green mask
            ## use cv2.addWeighted, segmentation is H x W, make it H x W x 3
            vis_segmentation = segmentation.reshape(segmentation.shape[0], segmentation.shape[1], 1)
            vis_segmentation = np.repeat(vis_segmentation, 3, axis=2)
            vis_segmentation = vis_segmentation.astype(np.uint8)
            vis_segmentation = vis_segmentation * 255

            ## save the image as human_name.png, convert from RGB to BGR
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)
            save_image_name = human_name+'.png'
            save_segmentation_name = 'seg_'+human_name+'.png'

            cv2.imwrite(save_image_name, temp_image)
            cv2.imwrite(save_segmentation_name, vis_segmentation)
            
        return segmentation
    
    def forward(self, positive_points, negative_points):
        
        positive_labels = np.ones(len(positive_points))
        negative_labels = np.zeros(len(negative_points))

        ## get low resolution segmentation using positive_points only
        input_points = positive_points
        input_labels = positive_labels

        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, self.image.shape[:2]).astype(np.float32)
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
                    "image_embeddings": self.image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(self.image.shape[:2], dtype=np.float32)
                }

        segmentation, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        segmentation = segmentation > self.predictor.model.mask_threshold ## 1 x 1 x H x W

        ## add negative_points
        if len(negative_points) > 0:
            input_points = np.concatenate([positive_points, negative_points], axis=0)
            input_labels = np.concatenate([positive_labels, negative_labels], axis=0)

            onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = self.predictor.transform.apply_coords(onnx_coord, self.image.shape[:2]).astype(np.float32)

            # Use the mask output from the previous run. It is already in the correct form for input to the ONNX model.
            onnx_mask_input = low_res_logits
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)

            ort_inputs = {
                    "image_embeddings": self.image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(self.image.shape[:2], dtype=np.float32)
                }

            segmentation, _, low_res_logits = self.ort_session.run(None, ort_inputs)
            segmentation = segmentation > self.predictor.model.mask_threshold ## 1 x 1 x H x W
        
        ##------------------------------------------------
        segmentation = segmentation[0][0]

        return segmentation
    
    def get_tight_bbox(self, pose):
        
        x1 = pose[:, 0].min(); x2 = pose[:, 0].max()
        y1 = pose[:, 1].min(); y2 = pose[:, 1].max()

        bbox_xyxy = np.array([x1, y1, x2, y2])

        return bbox_xyxy

    ####--------------------------------------------------------
    ## head_bboxes has identity
    ## prev_segmentation is the segmentation from previous frame with identity
    ## segmentation is the segmentation from current frame without identity
    def get_segmentation_association(self, head_bboxes, prev_segmentation, segmentation):

        ## sort the head_bboxes by 'distance_to_camera
        head_bboxes = sorted(head_bboxes, key=lambda x: x['distance_to_camera'])

        ## pick detection, compute scores, sort by scores then greedily match with gt
        head_scores = np.zeros((len(head_bboxes), len(segmentation))) ## gt x detections
        seg_scores = np.zeros((len(head_bboxes), len(segmentation))) ## gt x detections

        ## iou match between head_bboxes and segmentation
        for i, head_bbox in enumerate(head_bboxes):
            human_name = head_bbox['human_name']

            prev_human_segmentation = None

            if prev_segmentation is not None:
                prev_human_segmentation = prev_segmentation[human_name]['segmentation']

            for j, mask in enumerate(segmentation):
                head_scores[i][j] = self.iou_bbox_seg(head_bbox['head_bbox'], mask['segmentation'])
                
                if prev_human_segmentation is not None:
                    seg_scores[i][j] = self.iou_seg(prev_human_segmentation, mask['segmentation'])
        
        weight_head = 2 ## higher weight for head as it is a small bbox than segmentation
        weight_seg = 1
        
        scores = weight_head * head_scores + weight_seg * seg_scores

        ## create a list of matched masks
        head_bbox_indices, segmentation_indices = linear_sum_assignment(-scores) ## cost matrix is -scores

        final_segmentation = {}
        for i, j in zip(head_bbox_indices, segmentation_indices):
            head_bbox = head_bboxes[i]
            mask = segmentation[j]
            score = scores[i][j]

            if score == 0:
                head_bbox['segmentation'] = None
                head_bbox['bbox'] = None

                final_segmentation[human_name] = head_bbox
                continue

            human_name = head_bbox['human_name']
            head_bbox['segmentation'] = mask['segmentation']
            head_bbox['bbox'] = mask['bbox']

            final_segmentation[human_name] = head_bbox
        
        ## copy head_bboxes that are not matched
        for i, head_bbox in enumerate(head_bboxes):
            human_name = head_bbox['human_name']
            if human_name not in final_segmentation:
                head_bbox['segmentation'] = None
                head_bbox['bbox'] = None
                final_segmentation[human_name] = head_bbox
            
        return final_segmentation

    ####--------------------------------------------------------
    def iou_bbox_seg(self, bbox, mask):
        
        ## bbox is [x0, y0, x1, y1]
        ## mask is binary mask of size [h, w]
        ## computer iou between bbox and mask
        x0, y0, x1, y1, _ = bbox
        h, w = mask.shape

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        ## create binary mask for bbox, called mask_bbox, it is a boolean mask
        mask_bbox = np.zeros((h, w), dtype=np.bool)
        mask_bbox[y0:y1, x0:x1] = True

        intersection = np.logical_and(mask, mask_bbox)
        union = np.logical_or(mask, mask_bbox)
        iou = np.sum(intersection) / np.sum(union)

        return iou
    
    def iou_seg(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)

        return iou
    
    ####--------------------------------------------------------
    def get_segmentation_for_pose(self, image_name, bboxes):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        masks = {}

        for bbox_info in bboxes:
            bbox = bbox_info['bbox']
            positive_point = bbox_info['point_2d'].reshape(1, 2)
            human_name = bbox_info['human_name']

            negative_points = [bbox_info['point_2d'].reshape(1, 2) for bbox_info in bboxes if bbox_info['human_name'] != human_name]

            input_box = bbox[:4]
            input_points = np.concatenate([positive_point] + negative_points, axis=0)
            input_labels = np.array([1] + [0] * len(negative_points))

            mask, score, logit = self.predictor.predict(
                                point_coords=input_points,
                                point_labels=input_labels,
                                box=input_box,
                                multimask_output=True,
                            )
            ## pick mask with max score
            mask = mask[np.argmax(score)]
            masks[human_name] = {'bbox': bbox, 'mask': mask, 'color': bbox_info['color']}

        return masks
    
    def draw_segmentation(self, segmentation, image):
        canvas = np.zeros_like(image)
        for human_name in segmentation.keys():
            mask = segmentation[human_name]['segmentation'] ## remove batch dimension
            color = segmentation[human_name]['color']
            color = np.array([color[0], color[1], color[2], 0.6])

            if mask is not None:
                mask_image = np.concatenate([mask.reshape(mask.shape[0], mask.shape[1], 1)*color[0], \
                                                mask.reshape(mask.shape[0], mask.shape[1], 1)*color[1], \
                                                mask.reshape(mask.shape[0], mask.shape[1], 1)*color[2], \
                                                ], axis=2)
                canvas = np.maximum(canvas, mask_image)

        mask_image = 0.6 * image + 0.4 * canvas
        is_zero = np.sum(canvas, axis=2) == 0
        mask_image[is_zero] = image[is_zero]

        return mask_image

    def oks(self, keypoints1, keypoints2, keypoint_thres=0.3):
        is_valid = (keypoints1[:, 2] > keypoint_thres) * (keypoints2[:, 2] > keypoint_thres)

        ## probably different skeletons
        if is_valid.sum() == 0:
            return 0.1

        ###------------area------------------
        area1 = self.get_area_from_pose(pose=keypoints1, keypoint_thres=keypoint_thres)
        area2 = self.get_area_from_pose(pose=keypoints2, keypoint_thres=keypoint_thres)

        area = (area1 + area2)/2 ## average the bbox area

        ###------------compute oks------------------
        xg = keypoints1[:, 0]; yg = keypoints1[:, 1]
        xd = keypoints2[:, 0]; yd = keypoints2[:, 1]
        dx = xd - xg
        dy = yd - yg

        vars = (self.coco_17_sigmas * 2)**2
        e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2

        e = e[is_valid > 0]
        oks = np.sum(np.exp(-e)) / e.shape[0]

        return oks

    def distance(self, keypoints1, keypoints2, keypoint_thres=0.3):
        ## return the average euclidean distance between keypoints1 and keypoints2
        is_valid = (keypoints1[:, 2] > keypoint_thres) * (keypoints2[:, 2] > keypoint_thres)

        ## probably different skeletons
        if is_valid.sum() == 0:
            return 1000
        
        xg = keypoints1[:, 0]; yg = keypoints1[:, 1]
        xd = keypoints2[:, 0]; yd = keypoints2[:, 1]
        dx = xd - xg
        dy = yd - yg

        distance = np.sqrt(dx**2 + dy**2)

        distance = distance[is_valid > 0]
        distance = np.mean(distance)

        return distance
    
    ####--------------------------------------------------------
    def get_area_from_pose(self, pose, keypoint_thres=0.3):
        is_valid = pose[:, 2] > keypoint_thres
        
        x1 = pose[is_valid, 0].min(); x2 = pose[is_valid, 0].max()
        y1 = pose[is_valid, 1].min(); y2 = pose[is_valid, 1].max()

        area = (x2-x1)*(y2-y1) ## area of the bbox

        return area
    