from functools import cmp_to_key
import torch
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import numpy as np

CKPT_PATH = 'tools/lidar/models/pointpillars_kitti_202012221652utc.pth'
CFG_PATH = 'tools/lidar/configs/pointpillars_kitti.yml'


class PointCloudDetection():
    def __init__(self):
        kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
        self.visualizer = ml3d.vis.Visualizer()
        lut = ml3d.vis.LabelLUT()

        for val in sorted(kitti_labels.keys()):
            lut.add_label(kitti_labels[val], val)
        self.visualizer.set_lut("labels", lut)
        self.visualizer.set_lut("pred", lut)

        cfg = _ml3d.utils.Config.load_from_file(CFG_PATH)
        self.model = ml3d.models.PointPillars(**cfg.model)
        self.pipeline = ml3d.pipelines.ObjectDetection(self.model,
                                                       device='cuda',
                                                       split='test')
        # load the parameters.
        self.pipeline.load_ckpt(ckpt_path=CKPT_PATH)

    def detect_objects(self,
                       pcd_points: np.ndarray,
                       visualize: bool = False) -> list:
        '''
        CONFIG:
        pcd_points: Needs to be an array of shape N X 4 
        visualize: Indicate whether to display point clouds
        returns: A list of detected objects and its corresponding bounding boxes
        '''
        data = {
            'name': "my_point_cloud",
            'point': pcd_points,
            'calib': None,
            'feat': None,
            'label': None
        }
        # run inference on a single example.
        # returns dict with 'predict_labels' and 'predict_scores'.
        results = self.pipeline.run_inference(data)[0]
        if visualize:
            vis_data = {
                'name': 'my_point_cloud',
                "points": pcd_points,
                "labels": None,
            }
            self.visualizer.visualize([vis_data], bounding_boxes=results)
        return results