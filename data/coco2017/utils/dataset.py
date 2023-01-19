import os
import sys
import cv2
import math
import random
import matplotlib.pyplot as plt

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from data.coco2017.entity import JointType, params, HOICategory, ObjectCategory, Solution_Category
# coco dataset (coco_train, mode, imgids)
class HOIDataset(Dataset):
    def __init__(self, anno_read, insize, mode='train', task='liquid_tracking', n_samples=None):
        self.annos = anno_read
        assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        self.mode = mode
        self.task = task
        self.train_list = []
        self.test_list = []

        # according to the task you choose
        # load image lists to train or test
        if task == 'collaboration':
            if mode == 'train':
                f = open(params['coco_dir']+'/'+'train.txt', encoding='utf-8')
                for line in f:
                    self.train_list.append(line.strip())
                self.train_list = map(int, self.train_list)
                self.train_list = sorted(set(self.train_list))
                self.imgIds = self.train_list
                # obtain ids from image with HOIs, filter image without HOIs
                self.imgIds, self.hoi_imgIds = anno_read.getHOIIds(self.imgIds)
                self.imgIds, self.hoi_imgIds = sorted(set(self.imgIds)), sorted(set(self.hoi_imgIds))
            else:
                f = open(params['coco_dir']+'/'+'test.txt', encoding='utf-8')
                for line in f:
                    self.test_list.append(line.strip())
                self.test_list = map(int, self.test_list)
                self.test_list = sorted(set(self.test_list))
                self.imgIds = self.test_list
                # obtain ids from image with HOIs, filter image without HOIs
                self.imgIds, self.hoi_imgIds = anno_read.getHOIIds(self.imgIds)
                self.imgIds, self.hoi_imgIds = sorted(set(self.imgIds)), sorted(set(self.hoi_imgIds))

        if task == 'instance_segmentation' or task == 'liquid_tracking':
            self.num_class = 21
            if mode == 'train':
                f = open(params['coco_dir']+'/'+'train.txt', encoding='utf-8')
                for line in f:
                    self.train_list.append(line.strip())
                self.train_list = map(int, self.train_list)
                self.train_list = sorted(set(self.train_list))
                self.imgIds = self.train_list
                self.seg_imgIds = anno_read.getSegmentationIds(self.imgIds)
                self.seg_imgIds = sorted(set(self.seg_imgIds))
            else:
                f = open(params['coco_dir']+'/'+'test.txt', encoding='utf-8')
                for line in f:
                    self.test_list.append(line.strip())
                self.test_list = map(int, self.test_list)
                self.test_list = sorted(set(self.test_list))
                self.imgIds = self.test_list
                # obtain ids from image with HOIs, filter image without HOIs
                self.seg_imgIds = anno_read.getSegmentationIds(self.imgIds)
                self.seg_imgIds = sorted(set(self.seg_imgIds))


        # get path information of each frame
        self.frameIds, self.videoIds, self.date = anno_read.getFrameIds(self.imgIds)

        if self.mode in ['val', 'eval'] and n_samples is not None:
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))
        self.insize = insize

    def __len__(self):
        return len(self.imgIds)

    def get_hoi_annotation(self, ind=None, img_id=None):
        ''' get HOIs with their ID from imageID '''
        ''' get poses with their ID from imageID '''
        ''' get objects from imageID '''
        ''' load the image according to the directory information, if not return None'''

        if ind is not None:
            img_id = self.imgIds[ind]
            hoi_img_id = self.hoi_imgIds[ind]
        hoi_ids = self.annos.getHoiIds(hoi_imgIds=[hoi_img_id])       # identify ann ids of each frame
        pose_ids = self.annos.getPoseIds(pose_imgIds=[hoi_img_id])

        hois_for_img = self.annos.loadHois(img_ids=hoi_ids)
        poses_for_img = self.annos.loadPoses(img_ids=pose_ids)
        objects_for_img = self.annos.loadObjects(img_ids=[img_id])

        # if too few keypoints
        person_cnt = 0
        valid_poses_for_img = []
        for pose in poses_for_img:
            if pose['num_keypoints'] >= params['min_keypoints']:
                person_cnt += 1
                valid_poses_for_img.append(pose)

        # if person annotation
        if person_cnt > 0:
            poses_anns = valid_poses_for_img
        else:
            poses_anns = []
        hois_anns, obj_anns = hois_for_img, objects_for_img

        # get path information of each frame and load the image
        videoId = self.videoIds[ind]
        frameId = self.frameIds[ind]
        date = self.date[ind]
        if self.mode == 'train':
            # img_path = os.path.join(params['coco_dir'], '0802', videoId, '{:012d}.jpg'.format(frameId))
            img_path = params['coco_dir'] + '/' + 'BDAI_img' + '/' + 'collaboration' + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
        else:
            img_path = os.path.join(params['coco_dir'], 'val2017', '{:08d}.jpg'.format(img_id))
        img = cv2.imread(img_path)

        ignore_mask = np.zeros(img.shape[:2], 'bool')

        if self.mode == 'eval':
            return img, img_id, poses_for_img, hois_for_img, objects_for_img, ignore_mask
        return img, img_id, poses_anns, hois_anns, obj_anns, ignore_mask

    def parse_pose_annotation(self, poses_anns):
        '''transfer to array'''

        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for pose in poses_anns:
            ann_pose = np.array(pose['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(params['coco_joint_indices']):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and pose[0][JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = int(
                    (pose[0][JointType.LeftShoulder][0] + pose[0][JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = int(
                    (pose[0][JointType.LeftShoulder][1] + pose[0][JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2

            poses = np.vstack((poses, pose))

        #         gt_pose = np.array(ann['keypoints']).reshape(-1, 3)
        return poses

    def random_resize_img(self, img, ignore_mask, poses):
        '''data augmentation -- resize'''

        h, w, _ = img.shape
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(axis=1)**0.5

        min_scale = params['min_box_size']/bbox_sizes.min()
        max_scale = params['max_box_size']/bbox_sizes.max()

        min_scale = min(max(min_scale, params['min_scale']), 1)
        max_scale = min(max(max_scale, 1), params['max_scale'])

        scale = float((max_scale - min_scale) * random.random() + min_scale)
        shape = (round(w * scale), round(h * scale))
        resized_img, resized_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape)
        return resized_img, resized_mask, poses

    def random_rotate_img(self, img, mask, poses):
        ''' data augmentation -- rotate '''

        h, w, _ = img.shape
        # degree = (random.random() - 0.5) * 2 * params['max_rotate_degree']
        degree = np.random.randn() / 3 * params['max_rotate_degree']
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)), w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(img, R, (int(bbox[0]+0.5), int(bbox[1]+0.5)), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=[127.5, 127.5, 127.5])
        rotate_mask = cv2.warpAffine(mask.astype('uint8')*255, R, (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0

        tmp_poses = np.ones_like(poses)
        tmp_poses[:, :, :2] = poses[:, :, :2].copy()
        tmp_rotate_poses = np.dot(tmp_poses, R.T)  # apply rotation matrix to the poses
        rotate_poses = poses.copy()  # to keep visibility flag
        rotate_poses[:, :, :2] = tmp_rotate_poses
        return rotate_img, rotate_mask, rotate_poses

    def random_crop_img(self, img, ignore_mask, poses):
        ''' data augmentation -- crop '''

        h, w, _ = img.shape
        insize = self.insize
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox = random.choice(joint_bboxes)  # select a bbox randomly
        bbox_center = bbox[:2] + (bbox[2:] - bbox[:2])/2

        r_xy = np.random.rand(2)
        perturb = ((r_xy - 0.5) * 2 * params['center_perterb_max'])
        center = (bbox_center + perturb + 0.5).astype('i')

        crop_img = np.zeros((insize, insize, 3), 'uint8') + 127.5
        crop_mask = np.zeros((insize, insize), 'bool')

        offset = (center - (insize-1)/2 + 0.5).astype('i')
        offset_ = (center + (insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

        x1, y1 = (center - (insize-1)/2 + 0.5).astype('i')
        x2, y2 = (center + (insize-1)/2 + 0.5).astype('i')

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        x_from = -offset[0] if offset[0] < 0 else 0
        y_from = -offset[1] if offset[1] < 0 else 0
        x_to = insize - offset_[0] - 1 if offset_[0] >= 0 else insize - 1
        y_to = insize - offset_[1] - 1 if offset_[1] >= 0 else insize - 1

        crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
        crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[y1:y2+1, x1:x2+1].copy()

        poses[:, :, :2] -= offset
        return crop_img.astype('uint8'), crop_mask, poses

    def augment_data(self, img, ignore_mask, poses, bbox):
        ''' pose data augmentation '''

        aug_img = img.copy()
        aug_img, ignore_mask, poses, bbox = self.random_resize_img(aug_img, ignore_mask, poses, bbox)
        aug_img, ignore_mask, poses, bbox = self.random_rotate_img(aug_img, ignore_mask, poses, bbox)
        aug_img, ignore_mask, poses, bbox = self.random_crop_img(aug_img, ignore_mask, poses, bbox)
        if np.random.randint(2):
            aug_img = self.distort_color(aug_img)
        if np.random.randint(2):
            aug_img, ignore_mask, poses, bbox = self.flip_img(aug_img, ignore_mask, poses, bbox)

        return aug_img, ignore_mask, poses, bbox

    def resize_data(self, img, ignore_mask, poses, obj_bboxs, human_bbox, shape):
        """resize img, poses and bbox"""

        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), shape).astype('bool')
        if len(poses)>0:
            poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array((img_w, img_h)))
        objs = (obj_bboxs[:, :2] * np.array(shape) / np.array((img_w, img_h)))
        objs = np.hstack((objs, (obj_bboxs[:, 2:] * np.array(shape) / np.array((img_w, img_h)))))
        humans = (human_bbox[:, :2] * np.array(shape) / np.array((img_w, img_h)))
        humans = np.hstack((humans, (human_bbox[:, 2:] * np.array(shape) / np.array((img_w, img_h)))))
        return resized_img, ignore_mask, poses, objs, humans

    # return shape: (height, width)
    def generate_gaussian_heatmap(self, shape, joint, sigma):
        """ transform pose labels to heatmap """

        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        # generate the Gaussian distribution
        return gaussian_heatmap

    def generate_heatmaps(self, img, poses, heatmap_sigma):
        """ transform pose labels to heatmap """

        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose in poses:
                if pose[joint_index, 2] > 0:
                    jointmap = self.generate_gaussian_heatmap(img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        '''
        We take the maximum of the confidence maps instead of the average so that the precision of close by peaks remains distinct, 
        as illustrated in the right figure. At test time, we predict confidence maps and obtain body part candidates by performing non-maximum suppression.
        At test time, we predict confidence maps, and obtain body part candidates by performing non-maximum suppression.
        '''
        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        """ transform pose labels to paf """

        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector) # vertical component
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # grid_x, grid_y for going through each pixel
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width # paf_width : 8
        paf_flag = horizontal_paf_flag & vertical_paf_flag # combine two constraints
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)

        return constant_paf

    def generate_pafs(self, img, poses, paf_sigma):
        """ transform pose labels to paf """

        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma) #[2,368,368]
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0] # average
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def generate_pose_labels(self, img, poses, ignore_mask):
        """ transform pose labels to heatmap and paf """

        # img, ignore_mask, poses = self.augment_data(img, ignore_mask, poses)
        # resized_img, ignore_mask, resized_poses = self.resize_data(img, ignore_mask, poses, shape=(self.insize, self.insize))

        heatmaps = self.generate_heatmaps(img, poses, params['heatmap_sigma'])
        pafs = self.generate_pafs(img, poses, params['paf_sigma']) # params['paf_sigma']: 8
        ignore_mask = cv2.morphologyEx(ignore_mask.astype('uint8'), cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')
        return img, pafs, heatmaps, ignore_mask

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def parse_objects_annotation(self, obj_anns):
        """load bbox of objects and humans from annos"""

        count = 0

        for obj in obj_anns:
            obj_bbox = np.array(obj['object_bbox']).reshape(-1, 4)
            human_bbox = np.array(obj['human_bbox']).reshape(-1, 4)
            label = obj['object_category']
            count += 1
            if count == 1:
                objs = obj_bbox
                labels = label
                humans = human_bbox
            else:
                objs = np.vstack((objs, obj_bbox))
                humans = np.vstack((humans, human_bbox))
                labels.append(label)

        return objs, humans, labels

    def generate_hoi_labels(self, objs, labels, humans, hois_anns, hoi_nums=74):
        """load bbox of subject and object from resized bbox"""
        """generate one-hot verb_label from the hoi_category"""
        """generate one-hot verb_label from the hoi_category"""
        """generate pairs of subject and object from the connection"""

        obj_ids, verb_labels, sub_boxes, obj_boxes, sub_ids = [], [], [], [], []
        sub_obj_pairs = []
        count = 0
        for hoi in hois_anns:
            pairs = [hoi['connection']]
            for i in range(len(pairs)):
                pair = pairs[i]
                if pair in sub_obj_pairs:
                    action_id = HOICategory[hoi['hoi_category']]
                    verb_labels[sub_obj_pairs.index(pair)][action_id] = 1
                else:
                    sub_obj_pairs.append(pair)
                    obj_ids.append(pairs[i][1])
                    verb_label = [0 for _ in range(hoi_nums)]
                    action_id = HOICategory[hoi['hoi_category']]
                    verb_label[action_id] = 1
                    verb_labels.append(verb_label)
                    if pairs[i][0] not in sub_ids:
                        sub_ids.append(pairs[i][0])
                        sub_box = humans[pairs[i][0], :]
                        if count == 0:
                            sub_boxes = sub_box
                        else:
                            sub_boxes = np.vstack((sub_boxes, sub_box))
                    obj_box = objs[obj_ids[i], :]
                    if count == 0:
                        obj_boxes = obj_box
                        count += 1
                    else:
                        obj_boxes = np.vstack((obj_boxes, obj_box))


        obj_ids = torch.tensor(obj_ids)
        sub_obj_pairs = torch.tensor(sub_obj_pairs)
        verb_labels = torch.tensor(verb_labels)
        sub_boxes = torch.tensor(sub_boxes)
        obj_boxes = torch.tensor(obj_boxes)

        return sub_obj_pairs, verb_labels, sub_boxes, obj_boxes

    @property
    def classes(self):
        """Category names."""
        return ('Background','pipette','PCR_tube','tube','waste_box','vial','measuring_flask','beaker','wash_bottle',
                'water_bottle','erlenmeyer_flask','culture_plate','spoon','electronic_scale','LB_solution',
                'stopwatch','D_sorbitol','solution_P1','plastic_bottle','agarose','cell_spreader')
        
    def __len__(self):
        return len(self.seg_imgIds)
    
    def get_seg_annotation(self, ind=None, img_id=None, task = 'instance_segmentation'):
        ''' get segmentation from imageID '''
        ''' load the image according to the directory information '''

        if ind is not None:
            img_id = self.imgIds[ind]
            seg_img_id = self.seg_imgIds[ind]

        segs_for_img = self.annos.loadSegs(ids=[seg_img_id])
        segs_anns = segs_for_img

        # get path information of each frame and load the image
        videoId = self.videoIds[ind]
        frameId = self.frameIds[ind]
        date = self.date[ind]
        if self.mode == 'train':
            # img_path = os.path.join(params['coco_dir'], '0802', videoId, '{:012d}.jpg'.format(frameId))
            # img_path = params['coco_dir'] + '/' + 'BDAI_img' + '/' + task + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
            img_path = params['coco_dir'] + '/' + 'data' + '/' + 'img' + '/' + task + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
        else:
            # img_path = os.path.join(params['coco_dir'], 'val2017', '{:08d}.jpg'.format(img_id))
            img_path = params['coco_dir'] + '/' + 'data' + '/' + 'img' + '/' + task + '/' + date + '/' + videoId + '/' + '{:08d}.jpg'.format(frameId)
        img = cv2.imread(img_path)

        ignore_mask = np.zeros(img.shape[:2], 'bool')

        if self.mode == 'eval':
            return img, img_id, segs_anns
        return img, img_id, segs_anns

    def load_seg_annotation(self, segs_anns, task):
        ''' get object_polygon label from annos '''
        ''' get object_category label from annos '''
        ''' get solution_category label from annos '''

        for seg in segs_anns:
            # print("seg", seg)
            obj_id= seg['object_id']
            label = seg['object_category']
            object_polygon = seg['object_polygon']

            solution_id = []
            label_id = []
            for i in label:
                id = ObjectCategory[i]
                label_id.append(id)

            if task == 'liquid_tracking':
                solution_category = seg['solution_category']

                for i in solution_category:
                    liquid_id = Solution_Category[i[1]]
                    id = i[0]
                    id.append(liquid_id)
                    solution_id.append(id)


        # return obj_id, label_id, object_polygon, solution_id
        return obj_id, label_id, object_polygon

    def resize_data_seg(self, img, object_polygon, shape):
        """resize img, and polygon"""

        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)

        resized_polygon = []
        count = 0
        if len(object_polygon)>0:
            for obj in object_polygon:
                if count == 0:
                    np_obj = np.array(obj).reshape(-1, 2) * np.array(shape) / np.array((img_w, img_h))
                    np_objs = np_obj.flatten()
                    np_objs = [np_objs.tolist()]
                    count += 1
                else:
                    np_obj = np.array(obj).reshape(-1, 2) * np.array(shape) / np.array((img_w, img_h))
                    np_obj = np_obj.flatten()
                    np_obj = [np_obj.tolist()]
                    # np_objs = np.vstack((np_objs, np_obj))
                    np_objs = np_objs + np_obj
                    count += 1

            resized_polygon = np_objs

        return resized_img, resized_polygon


    def annToRLE(self, polygon, h, w):
        """
        transform polygon to feature map mask
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = [polygon]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = polygon
        return rle

    def annToMask(self, polygon, h, w):
        """
        transform polygon to feature map mask
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        # print("polygon shape", len(polygon))
        rle = self.annToRLE(polygon, h, w)
        # print("rle shape", rle)
        m = maskUtils.decode(rle)
        # print("m shape", m.shape)
        return m

    def parse_seg_annotation(self, resized_polygon, img):
        """transform polygon to feature map mask"""

        img_h, img_w, _ = img.shape
        # print("len", len(resized_polygon[0]), len(resized_polygon[1]))

        masks = [self.annToMask(polygon, img_h, img_w).reshape(-1) for polygon in resized_polygon]
        masks = np.vstack(masks)
        masks = masks.reshape(-1, img_h, img_w)

        return masks

    def __getitem__(self, i):
        if self.task == 'collaboration':
            img, img_id, poses_anns, hois_anns, obj_anns, ignore_mask = self.get_hoi_annotation(ind=i)
            if self.mode == 'eval':
                # don't need to make heatmaps/pafs
                return img, poses_anns, hois_anns, obj_anns, img_id
            if len(poses_anns) > 0:
                poses = self.parse_pose_annotation(poses_anns)
            else:
                poses = []
            obj_bboxs, human_bbox, labels = self.parse_objects_annotation(obj_anns)
            # img, ignore_mask, poses, bbox = self.augment_data(img, ignore_mask, poses, bbox)
            resized_img, ignore_mask, resized_poses, resized_obj_bbox, resized_human_bbox = self.resize_data(img,
                                                                                                             ignore_mask,
                                                                                                             poses,
                                                                                                             obj_bboxs,
                                                                                                             human_bbox,
                                                                                                             shape=(
                                                                                                             self.insize,
                                                                                                             self.insize))

            sub_obj_pairs, verb_labels, hoi_sub_boxes, hoi_obj_boxes = self.generate_hoi_labels(resized_obj_bbox,
                                                                                                labels,
                                                                                                resized_human_bbox,
                                                                                                hois_anns)

            resized_img, pafs, heatmaps, ignore_mask = self.generate_pose_labels(resized_img, resized_poses,
                                                                                 ignore_mask)
            resized_img = self.preprocess(resized_img)
            resized_img = torch.tensor(resized_img)
            pafs = torch.tensor(pafs)
            heatmaps = torch.tensor(heatmaps)
            ignore_mask = torch.tensor(ignore_mask.astype('f'))

            resized_obj_bbox = torch.tensor(resized_obj_bbox)
            labels = torch.tensor(labels)
            return resized_img, pafs, heatmaps, ignore_mask, resized_obj_bbox, labels, sub_obj_pairs, verb_labels, hoi_sub_boxes, hoi_obj_boxes

        if self.task == 'instance_segmentation':
            img, img_id, segs_anns = self.get_seg_annotation(ind=i, task=self.task)

            obj_ids, labels, object_polygon = self.load_seg_annotation(segs_anns, task=self.task)
            resized_img, resized_polygon = self.resize_data_seg(img, object_polygon, shape=(self.insize, self.insize))
            marks = self.parse_seg_annotation(resized_polygon, resized_img)

            marks = torch.tensor(marks)
            labels = torch.tensor(labels)
            obj_ids = torch.tensor(obj_ids)
            
            # print("1", type(resized_img), resized_img.shape)
            # print("2", marks.shape[0])
            # print("2", type(marks))
            # np.savetxt("ttt.txt", marks[1].numpy().astype(np.int32))
            # for i in range(7):
            #     f = np.array(marks[i]).astype(np.int32)
            #     plt.imshow(f)
            #     plt.show()
            # print("3", labels.size())
            # print("4", obj_ids.size())
            # print("obj", obj_ids)
            # print("labels", labels)
            
            for i in range(marks.shape[0]):
                marks[i]*=labels[i]
                
            marks=marks.sum(dim=0)
            # print(type(marks.sum(dim=0)))
            # print("sss", marks.shape)
            # resized_img = resized_img.resize(3,512,512)

            return resized_img, marks, labels, obj_ids

        if self.task == 'liquid_tracking':
            img, img_id, segs_anns = self.get_seg_annotation(ind=i, task=self.task)

            obj_ids, labels, object_polygon, solution_ids = self.load_seg_annotation(segs_anns, task=self.task)
            resized_img, resized_polygon = self.resize_data_seg(img, object_polygon, shape=(self.insize, self.insize))
            marks = self.parse_seg_annotation(resized_polygon, resized_img)

            marks = torch.tensor(marks)
            labels = torch.tensor(labels)
            obj_ids = torch.tensor(obj_ids)
            solution_ids = torch.tensor(solution_ids)

            return resized_img, marks, labels, obj_ids, solution_ids







