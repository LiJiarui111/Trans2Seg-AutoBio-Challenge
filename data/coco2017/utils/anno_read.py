import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys

# sys.path.append("..")
from data.coco2017.entity import JointType, params
# from .. import entity
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class anno_read:
    def __init__(self, annotation_file=None, mode='train'):
        self.dataset, self.imgs, self.poses, self.hois, self.objects, self.segs = dict(), dict(), dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            print("annotation file path: ", annotation_file)
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()


    def createIndex(self):
        # create index of img, pose, hois, objects and segmentation
        print('creating index...')
        poses, hois, imgs, objects = {}, {}, {}, {}
        imgToPoses, hoiToImgs, catToImgs = defaultdict(list), defaultdict(list), defaultdict(list)
        if 'pose' in self.dataset:
            # print("1")
            for pose in self.dataset['pose']:
                imgToPoses[pose['image_id']].append(pose)
                poses[pose['id']] = pose

        if 'images' in self.dataset:
            # print("2")
            for img in self.dataset['images']:
                imgs[img['image_id']] = img

        if 'hoi' in self.dataset:
            # print("3")
            for hoi in self.dataset['hoi']:
                hoiToImgs[hoi['image_id']].append(hoi)
                hois[hoi['id']] = hoi

        if 'objects' in self.dataset:
            # print("4")
            for obj in self.dataset['objects']:
                objects[obj['image_id']] = obj

        if 'segmentation' in self.dataset:
            # print("5")
            for seg in self.dataset['segmentation']:
                self.segs[seg['image_id']] = seg

        # time.sleep(3)
        print('index created!')

        # create class members
        self.poses = poses
        self.imgToPoses = imgToPoses
        self.hois = hois
        self.hoiToImgs = hoiToImgs
        self.imgs = imgs
        self.objects = objects
        # self.catToImgs = catToImgs

    def getSegmentationIds(self, img_list=[], is_tracking=0):
        """
        obtian seg_ids from given train/test list
        """
        seg_id = []
        for seg in self.dataset['segmentation']:
            if seg['image_id'] in img_list:
                seg_id.append(seg['image_id'])

        return seg_id

    def getHOIIds(self, img_list=[]):
        """
        1. filter out image without HOIs
        2. obtian hoi_ids from given train/test list
        """
        img_ids, all_hoi, hoi_ids = [], [], []

        # filter out image without HOIs
        for hoi in self.dataset['hoi']:
            all_hoi.append(hoi['image_id'])
        for i in img_list:
            if i in all_hoi:
                img_ids.append(i)

        # obtian hoi_ids from given train/test list
        for hoi in self.dataset['hoi']:
            if hoi['image_id'] in img_ids:
                hoi_ids.append(hoi['image_id'])

        return img_ids, hoi_ids

    def getFrameIds(self, img_id):
        """
        the image location from imageid
        the directory is composed of date, frame_ids, video_ids
        """
        frame_ids, video_ids, date = [], [], []
        for img in self.dataset['images']:
            if img['image_id'] in img_id:
                frame_ids.append(img['frame_id'])
                video_ids.append(img['video_id'])
                date.append(img['date'])

        return frame_ids, video_ids, date


    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[0][catId])
                else:
                    ids &= set(self.catToImgs[0][catId])
        return list(ids)

    # def getHoiIds(self, hoi_imgIds=[]):
    #     """
    #     get HOIids from the given img_id
    #     """
    #     imgIds = hoi_imgIds if _isArrayLike(hoi_imgIds) else [hoi_imgIds]
    #
    #     if len(imgIds) == 0:
    #         hois = self.dataset['hoi']
    #     else:
    #         if not len(imgIds) == 0:
    #             lists = [self.hoiToImgs[imgId] for imgId in imgIds if imgId in self.hoiToImgs]
    #             hois = list(itertools.chain.from_iterable(lists))
    #         else:
    #             hois = self.dataset['hoi']
    #
    #     ids = [hoi['id'] for hoi in hois]
    #
    #     return ids

    def getPoseIds(self, pose_imgIds=[]):
        """
        get poseids from the given img_id
        """
        imgIds = pose_imgIds if _isArrayLike(pose_imgIds) else [pose_imgIds]

        if len(imgIds) == 0:
            poses = self.dataset['pose']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToPoses[imgId] for imgId in imgIds if imgId in self.imgToPoses]
                poses = list(itertools.chain.from_iterable(lists))
            else:
                poses = self.dataset['pose']
            poses = poses


        ids = [pose['id'] for pose in poses]

        return ids

    def loadHois(self, img_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(img_ids):
            return [self.hois[id] for id in img_ids]
        elif type(img_ids) == int:
            return [self.hois[id] for id in img_ids]

    def loadPoses(self, img_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(img_ids):
            return [self.poses[id] for id in img_ids]
        elif type(img_ids) == int:
            return [self.poses[id] for id in img_ids]

    def loadObjects(self, img_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(img_ids):
            return [self.objects[id] for id in img_ids]
        elif type(img_ids) == int:
            return [self.objects[id] for id in img_ids]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


    def loadSegs(self, ids=[]):
        """
        load segmentation annos from the given img_id
        """
        if _isArrayLike(ids):
            return [self.segs[id] for id in ids]
        elif type(ids) == int:
            return [self.segs[id] for id in ids]
        
if __name__ == "__main__":
    # af_path = 
    cur_path = os.path.abspath(os.path.dirname(__file__))
    print(cur_path)

