from PointCloudDetection import PointCloudDetection
import open3d as o3d
import numpy as np

detector = PointCloudDetection()
pcd = o3d.io.read_point_cloud("test_data/lidar/2050_frame_414747.426105.pcd")
points = np.array(pcd.points).astype(np.float32)
points = np.c_[points, np.zeros((points.shape[0], 1))].astype(np.float32)
results = detector.detect_objects(points, visualize=True)
print(results)