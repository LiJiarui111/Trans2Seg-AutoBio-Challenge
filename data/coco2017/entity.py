from enum import IntEnum

from data.coco2017.models.CocoPoseNet import CocoPoseNet

from data.coco2017.models.FaceNet import FaceNet
from data.coco2017.models.HandNet import HandNet
from pathlib import Path

class JointType(IntEnum):
    """ types of keypoints """
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17

HOICategory = {
    """ types of HOIs """
    """ composed of object and action """
    
    'total': 74,
    'pipette_inject': 1,
    'pipette_insert': 2,
    'pipette_hold': 3,
    'pipette_draw': 4,
    'pipette_take': 5,
    'pipette_put': 6,
    'pipette_discard': 7,
    'PCR_tube_hold': 8,
    'PCR_tube_take': 9,
    'PCR_tube_shake': 10,
    'PCR_tube_put': 11,
    'PCR_tube_flick': 12,
    'tube_hold': 13,
    'tube_take': 14,
    'tube_put': 15,
    'tube_open': 16,
    'tube_close': 17,
    'tube_shake': 18,
    'waste_box_take': 19,
    'waste_box_put': 20,
    'waste_box_hold': 21,
    'erlenmeyer_flask_hold': 22,
    'erlenmeyer_flask_put': 23,
    'erlenmeyer_flask_take': 24,
    'erlenmeyer_flask_shake': 25,
    'erlenmeyer_flask_seal': 26,
    'erlenmeyer_flask_open': 27,
    'agarose_hold': 28,
    'agarose_open': 29,
    'agarose_close': 30,
    'agarose_take': 31,
    'agarose_pour': 32,
    'spoon_scoop': 33,
    'spoon_pour': 34,
    'spoon_hold': 35,
    'spoon_put': 36,
    'LB_solution_shake': 37,
    'LB_solution_take': 38,
    'LB_solution_hold': 39,
    'LB_solution_put': 40,
    'LB_solution_seal': 41,
    'LB_solution_open': 42,
    'stopwatch_set': 43,
    'stopwatch_take': 44,
    'stopwatch_hold': 45,
    'stopwatch_put': 46,
    'cell_spreader_take': 47,
    'cell_spreader_open': 48,
    'cell_spreader_daub': 49,
    'cell_spreader_discard': 50,
    'culture_plate_hold': 51,
    'culture_plate_take': 52,
    'culture_plate_put': 53,
    'solution_P1_take': 54,
    'solution_P1_put': 55,
    'solution_P1_hold': 56,
    'solution_P1_open': 57,
    'solution_P1_close': 58,
    'measuring_flask_take': 59,
    'measuring_flask_put': 60,
    'measuring_flask_pour': 61,
    'measuring_flask_hold': 62,
    'vial_take': 63,
    'vial_put': 64,
    'vial_hold': 65,
    'vial_open': 66,
    'vial_close': 67,
    'vial_shake': 68,
    'D_sorbitol_open': 69,
    'D_sorbitol_close': 70,
    'D_sorbitol_hold': 71,
    'D_sorbitol_take': 72,
    'D_sorbitol_put': 73,
    'D_sorbitol_pour': 74,
}

ObjectCategory = {
    """ types of objects """
    
    'pipette':1,
    'PCR_tube':2,
    'tube':3,
    'waste_box':4,
    'vial':5,
    'measuring_flask':6,
    'beaker':7,
    'wash_bottle':8,
    'water_bottle':9,
    'erlenmeyer_flask':10,
    'culture_plate':11,
    'spoon':12,
    'electronic_scale':13,
    'LB_solution':14,
    'stopwatch':15,
    'D_sorbitol':16,
    'solution_P1':17,
    'plastic_bottle':18,
    'agarose':19,
    'cell_spreader':20,
}

Solution_Category = {
    """ types of solution """
    
    'none':0,
    'bacteria':1,
    'solution_P1':2,
    'LB_solution':3,
    'dd_water':4,
}

params = {
    'coco_dir': './data/coco2017',
    'archs': {
        'posenet': CocoPoseNet,
        'facenet': FaceNet,
        'handnet': HandNet,
    },
    'pretrained_path' : 'models/pretrained_vgg_base.pth',
    # training params
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    'batch_size': 2,
    'lr': 1e-4,
    'num_workers': 0,
    'eva_num': 100,
    'board_loss_interval': 100,
    'eval_interval': 4,
    'board_pred_image_interval': 2,
    'save_interval': 2,
    'log_path': 'work_space/log',
    'work_space': Path('work_space'),
    
    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'coco_joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ],

    # face params
    'face_inference_img_size': 368,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
    'face_line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20], [20, 21],
        [22, 23], [23, 24], [24, 25], [25, 26],
        [27, 28], [28, 29], [29, 30],
        [31, 32], [32, 33], [33, 34], [34, 35],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
    ],

    # hand params
    'hand_inference_img_size': 368,
    'hand_heatmap_peak_thresh': 0.1,
    'fingers_indices': [
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 5], [5, 6], [6, 7], [7, 8]],
        [[0, 9], [9, 10], [10, 11], [11, 12]],
        [[0, 13], [13, 14], [14, 15], [15, 16]],
        [[0, 17], [17, 18], [18, 19], [19, 20]],
    ],
}
