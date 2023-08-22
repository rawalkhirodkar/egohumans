import os
import numpy as np
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import cv2
import concurrent.futures
from multiprocessing import Pool
from mmpose.analysis.vis_utils import vis_keypoints, vis_bbox

# ------------------------------------------------------------------
def check_valid_annotations(coco, image_id):
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # ----------------------
    # sanitize bboxes
    image_info = coco.loadImgs(image_id)[0]
    width = image_info['width']
    height = image_info['height']
    valid_objs = []
    for obj in annotations:
        # ignore objs without keypoints annotation
        if max(obj['keypoints']) == 0:
            continue
        x, y, w, h = obj['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
            valid_objs.append(obj)
    annotations = valid_objs

    # ------------------------------------------------
    valid_annotation_ids = []
    valid_image_ids = []

    for annotation_idx, annotation in enumerate(annotations):
	    valid_annotation_ids.append(annotation['id'])

    if len(valid_annotation_ids) > 0:
    	valid_image_ids.append(image_id)

    return valid_annotation_ids, valid_image_ids

# ------------------------------------------------------------------
def sort_image_ap(gt_file, dt_file, output_dir):
	# gt_file = 'data/coco/annotations/person_keypoints_val2017.json'
	# image_dir = 'data/coco/val2017'

	image_dir = os.path.join(os.path.dirname(os.path.dirname(gt_file)), 'val2017')

	coco_gt = COCO(gt_file)
	coco_dt = coco_gt.loadRes(dt_file)

	info_str = print_evaluation(coco_gt, coco_dt)

	save_image_dir = os.path.join(output_dir, 'images')
	if not os.path.exists(save_image_dir):
		os.makedirs(save_image_dir)

	# -------------------------------------------
	image_ids = coco_gt.getImgIds()
	all_annotation_ids = [] ## 6352 person annotations
	all_image_ids = [] ## total 2346 person images

	for image_id in image_ids:
		valid_image_ann_ids, valid_image_ids = check_valid_annotations(coco_gt, image_id)
		all_annotation_ids += valid_image_ann_ids
		all_image_ids += valid_image_ids



	# # -------------------------------------------
	# performance_dict = {} ## {image_path: AP}
	
	# ## use tqdm to show progress bar
	# for idx, image_id in enumerate(tqdm(all_image_ids)):
		
	# 	image_name = coco_gt.loadImgs(image_id)[0]['file_name']
	# 	image_path = os.path.join(image_dir, image_name)

	# 	image_performance_info_str, gt_annotations, dt_annotations = image_evaluate(coco_gt, coco_dt_file=dt_file, image_id=image_id)

	# 	print('Done evaluation for image_id:{}. AP:{}'.format(image_id, round(image_performance_info_str['AP'], 3)))

	# 	performance_dict[image_id] = {'AP': image_performance_info_str['AP'], 'image_path': image_path,\
	# 				 'dt_annotations': dt_annotations, 'gt_annotations': gt_annotations}
		
	# # -------------------------------------------
	performance_dict = {} ## {image_path: AP}

	## use multiprocessing pool to parallelize image_evaluate
	with Pool(processes=os.cpu_count()) as pool:
		all_results = pool.map(image_evaluate, [(coco_gt, dt_file, image_id) for image_id in all_image_ids])

	## use tqdm to show progress bar
	for idx, (image_info_str, gt_annotations, dt_annotations) in enumerate(tqdm(all_results)):
		image_id = all_image_ids[idx]
		image_name = coco_gt.loadImgs(image_id)[0]['file_name']
		image_path = os.path.join(image_dir, image_name)

		print('Done evaluation for image_id:{}. AP:{}'.format(image_id, round(image_info_str['AP'], 3)))

		performance_dict[image_id] = {'AP': image_info_str['AP'], 'image_path': image_path,\
                      'dt_annotations': dt_annotations, 'gt_annotations': gt_annotations}

	##-------------------------------------------
	## sort the performance_dict by AP
	sorted_performance_dict = sorted(performance_dict.items(), key=lambda x: x[1]['AP'], reverse=False)

	## save the sorted_performance_dict
	sorted_performance_dict_file = os.path.join(output_dir, 'sorted_performance_dict.npy')
	np.save(sorted_performance_dict_file, sorted_performance_dict)

	## iterate through the sorted_performance_dict and save the images
	for idx, (image_id, image_info) in enumerate(sorted_performance_dict):
		print('Saving image: {} with AP: {}'.format(image_info['image_path'], image_info['AP']))

		## load the image
		image = cv2.imread(image_info['image_path'])

		gt_image = image.copy()
		dt_image = image.copy()

		## draw the gt_annotations
		for gt_annotation in image_info['gt_annotations']:

			gt_keypoints = np.array(gt_annotation['keypoints'])
			gt_kps = np.zeros((3, 17)).astype(np.int16)
			gt_kps[0, :] = gt_keypoints[0::3]; 
			gt_kps[1, :] = gt_keypoints[1::3]
			gt_kps[2, :] = gt_keypoints[2::3]

			gt_image = vis_keypoints(img=gt_image, kps=gt_kps.copy(), kp_thresh=-1, alpha=0.95)
			gt_image = vis_bbox(img=gt_image, bbox=gt_annotation['clean_bbox'])

		## draw the dt_annotations
		for dt_annotation in image_info['dt_annotations']:
			dt_keypoints = np.array(dt_annotation['keypoints'])

			dt_conf = dt_keypoints[2::3]

			dt_kps = np.zeros((3, 17)).astype(np.int16)
			dt_kps[0, :] = dt_keypoints[0::3]; 
			dt_kps[1, :] = dt_keypoints[1::3]

			## convert to binary, if dt_conf > 0.5
			dt_kps[2, :] = (dt_conf > 0.5).astype(np.int16)

			dt_image = vis_keypoints(img=dt_image, kps=dt_kps.copy(), kp_thresh=-1, alpha=0.95)
			dt_image = vis_bbox(img=dt_image, bbox=dt_annotation['bbox'])
		
		## save the images
		white_separator = np.ones((image.shape[0], 5, 3)).astype(np.uint8)*255
		save_image = np.concatenate((gt_image, white_separator, dt_image), axis=1)

		## name the save image such that lower AP images are saved first,
		performance_AP = round(1000*image_info['AP'])
		performance_AP_str = str(performance_AP).zfill(4)

		save_image_name = 'AP_{}_id_{}.jpg'.format(performance_AP_str, image_id)

		save_image_path = os.path.join(save_image_dir, save_image_name)
		cv2.imwrite(save_image_path, save_image)


	return 

# ------------------------------------------------------------------
def image_evaluate(args):

	coco_gt, coco_dt_file, image_id = args

	## create a new coco_gt with only one image
	image_coco_gt = copy.deepcopy(coco_gt)
	image_coco_dt = image_coco_gt.loadRes(coco_dt_file)

	# -------------------------------------------
	image_coco_gt.dataset['annotations'] = [coco_gt.anns[id] for id in image_coco_gt.getAnnIds(imgIds=image_id)]
	image_coco_gt.imgs = {image_id: coco_gt.imgs[image_id]}
	image_coco_gt.createIndex() ## create index again

	# -------------------------------------------
	## now trim coco_dt as well
	valid_dt_annotations = []
	for annotation in image_coco_dt.dataset['annotations']:
		if annotation['image_id'] == image_id:
			valid_dt_annotations.append(annotation)
	
	image_coco_dt.dataset['annotations'] = valid_dt_annotations
	image_coco_dt.imgs = {image_id: image_coco_dt.imgs[image_id]}
	image_coco_dt.createIndex() ## create index again

	image_info_str = print_evaluation(image_coco_gt, image_coco_dt, print=False)
        
	## filter out the detections with score < 0.05
	dt_annotations = []
	for annotation in valid_dt_annotations:
		if annotation['score'] > 0.05:
			dt_annotations.append(annotation)
                
	## only pick category_id = 1 in gt_annotations
	gt_annotations = []
	for annotation in image_coco_gt.dataset['annotations']:
		if annotation['category_id'] == 1:
			## pop off segmentation
			annotation.pop('segmentation', None)
			gt_annotations.append(annotation)
	
	# ----------------------
	# sanitize bboxes
	image_info = coco_gt.loadImgs(image_id)[0]
	width = image_info['width']
	height = image_info['height']
	valid_objs = []
	for obj in gt_annotations:
		# ignore objs without keypoints annotation
		if max(obj['keypoints']) == 0:
			continue
		x, y, w, h = obj['bbox']
		x1 = np.max((0, x))
		y1 = np.max((0, y))
		x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
		y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
		if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
			obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
			valid_objs.append(obj)
	gt_annotations = valid_objs

	return image_info_str, gt_annotations, dt_annotations

# ------------------------------------------------------------------
def summarize_oks(coco_eval, ap=1, iouThr=0.85, areaRng='all', maxDets=20):
    p = coco_eval.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap==1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s

# ------------------------------------------------------------------
def print_evaluation(coco_gt, coco_dt, print=True):
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
	coco_eval.params.useSegm = None
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

	info_str = {}
	for ind, name in enumerate(stats_names):
		info_str[name] = coco_eval.stats[ind]

	return info_str


















