import pandas as pd
import numpy as np
from tools.utils import CONFIG


def get_transform_matrix(file):

    clb = {}
    with open(file, 'r') as f:
        for line in f:
            calib_line = line.split(':')
            if len(calib_line) < 2:
                continue
            key = calib_line[0]
            value = np.array(list(map(float, calib_line[1].split())))
            value = value.reshape((3, -1))
            clb[key] = value

    P2 = clb['P2']
    R0 = np.eye(4)
    R0[:-1, :-1] = clb['R0_rect']
    Tr = np.eye(4)
    Tr[:-1, :] = clb['Tr_velo_to_cam']
    return P2, R0, Tr


def get_pointcloud(file):

    points = np.fromfile(file, dtype=np.float32)
    points = points.reshape((-1, 4)).T
    points = points[:, points[2, :] >= CONFIG['z-range'][0] + 0.3].copy()
    point_df = pd.DataFrame.from_records(np.transpose(points),
                                         columns=['x', 'y', 'z', 'l'])
    point_df['z_dist'] = np.sqrt(point_df['x']**2 + point_df['y']**2)
    point_df['o'] = np.degrees(np.arctan(point_df['z'] / point_df['z_dist']))
    return point_df, points
